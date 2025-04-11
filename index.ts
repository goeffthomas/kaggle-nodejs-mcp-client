
import { FunctionDeclaration, GenerativeModel, GoogleGenerativeAI, Schema, SchemaType, FunctionDeclarationsTool, ChatSession } from "@google/generative-ai";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import readline from "readline/promises";
import dotenv from "dotenv";
import { ZodArray, ZodBoolean, ZodDefault, ZodEnum, ZodNullable, ZodNumber, ZodObject, ZodOptional, ZodString, ZodTypeAny } from 'zod';

// Ensure you have a valid GEMINI_API_KEY in your `.env` file
dotenv.config();

const SERVER_PATHS = ["/git/kaggle-nodejs-mcp-server/build/index.js"];
const SYSTEM_PROMPT = `You are an expert data analyst. Use the tools available to you to answer the user's questions. 

If the user asks you to analyze a specific dataset (either to answer questions or generate a python notebook), you should fetch the metadata 
for the provided dataset in order to answer the question.

References to datasets and notebooks can be in the form of fully qualified URLs like https://www.kaggle.com/<resource>/<owner_slug>/<resource_slug> 
or simply as a handle in the form of <owner_slug>/<resource_slug>.

When generating a python notebook, keep in mind that all datasets are mounted in a "/kaggle/input" directory. For example, a notebook that has a 
dataset called "my-amazing-dataset", with a "awesome-data.csv" inside would have the file located at "/kaggle/input/my-amazing-dataset/awesome-data.csv". 
Ensure that all references to files fit this pattern.`;

class MCPClient {
    private mcp: Client;
    private gemini: GenerativeModel;
    private session: ChatSession;
    private transport: StdioClientTransport | null = null;
    private tools: FunctionDeclarationsTool[] = [];

    constructor() {
        const { GEMINI_API_KEY } = process.env;
        if (!GEMINI_API_KEY) {
            throw new Error("GEMINI_API_KEY is not set");
        }
        this.gemini = new GoogleGenerativeAI(GEMINI_API_KEY).getGenerativeModel({ model: "gemini-2.0-flash" });
        this.mcp = new Client({ name: "mcp-client-cli", version: "1.0.0" });
        this.session = {} as ChatSession;
    }


    async connectToServer(serverScriptPath: string) {
        try {
            const isJs = serverScriptPath.endsWith(".js");
            const isPy = serverScriptPath.endsWith(".py");
            if (!isJs && !isPy) {
                throw new Error("Server script must be a .js or .py file");
            }
            const command = isPy
                ? process.platform === "win32"
                    ? "python"
                    : "python3"
                : process.execPath;
            
            this.transport = new StdioClientTransport({
                command,
                args: [serverScriptPath],
            });
            this.mcp.connect(this.transport);
            
            this.tools = await processTools(this.mcp);
            this.session = this.gemini.startChat({
                systemInstruction: { role: "system", parts: [{text: SYSTEM_PROMPT }]},
                tools: this.tools,
            });
            console.log(
                "Connected to server with tools:",
                this.tools.map(t => t.functionDeclarations?.[0].name)
            );
        } catch (e) {
            console.log("Failed to connect to MCP server: ", e);
            throw e;
        }
    }

    async processQuery(query: string) {
        const result = await this.session.sendMessage(query);
      
        const finalText = [result.response.text()];
        for (const functionCall of result.response.functionCalls() ?? []) {
            const toolName = functionCall.name;
            const toolArgs = functionCall.args as { [k: string]: unknown };
    
            finalText.push(
                `[Calling tool ${toolName} with args ${JSON.stringify(toolArgs)}]`
              );
            const result = await this.mcp.callTool({
              name: toolName,
              arguments: toolArgs,
            });

            console.log(result)
      
            // Using any is not great, but these types are really hard to work with
            const geminiResult = await this.session.sendMessage(`Using the following information, answer the original query: ${(result.content as any)[0].text}`);
      
            finalText.push(geminiResult.response.text());
        }
      
        return finalText.join("\n");
    }

    async chatLoop() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });
        
        try {
            console.log("\nMCP Client Started!");
            console.log("Type your queries or 'quit' to exit.");
        
            while (true) {
                const message = await rl.question("\nQuery/Response: ");
                if (message.toLowerCase().trim() === "quit") {
                    break;
                }
                const response = await this.processQuery(message);
                console.log("\n" + response);
            }
        } catch (err: any) {
            console.error(err);
        } finally {
            rl.close();
        }
    }
        
    async cleanup() {
        await this.mcp.close();
    }
}

// Convert discovered tools into Google tools for Gemini
async function processTools(mcp: Client): Promise<FunctionDeclarationsTool[]> {
    const toolsResult = await mcp.listTools();

    const googleTools: FunctionDeclarationsTool[] = toolsResult.tools.map(tool => {
        const googleProperties: { [k: string]: Schema } = {};
        const inputProperties = tool.inputSchema.properties; // This is the ZodPropertiesMap

        // Iterate over the properties map which contains Zod types as values
        if (inputProperties) {
            for (const key in inputProperties) {
                if (Object.prototype.hasOwnProperty.call(inputProperties, key)) {
                    googleProperties[key] = zodTypeToGoogleSchema(inputProperties[key] as ZodTypeAny); // Convert it
                }
            }
        }

        // Construct the final parameters schema for Google
        const googleParameters: Schema = {
            type: SchemaType.OBJECT, // Top level must be object
            properties: googleProperties,
            description: tool.description // Use description if available
        };

        const funcDecl: FunctionDeclaration = {
            name: tool.name,
            description: tool.description,
            parameters: googleParameters
        };

        return { functionDeclarations: [funcDecl] };
    });

    return googleTools;
}

// --- Helper Function: Convert Zod Type to Google Schema ---
// NOTE: This was all created by Gemini with some minor tweaks to satisfy TS.
// This was the worst part of reworking Anthropic's examples to fit Gemini.
function zodTypeToGoogleSchema(zodType: ZodTypeAny): Schema {
    // --- Handle Zod wrappers first (Optional, Nullable, Default) ---
    if (zodType instanceof ZodOptional || zodType instanceof ZodNullable) {
        // Google's schema infers optionality from the 'required' array at the object level.
        // Nullable isn't directly supported in basic JSON schema for Gemini, often handled via description or union types (if supported).
        zodType = zodType.unwrap(); // Get the inner type
    }
    if (zodType instanceof ZodDefault) {
        zodType = zodType._def.innerType; // Get the inner type
    }
     // Re-check for optional/nullable after unwrapping default
    if (zodType instanceof ZodOptional || zodType instanceof ZodNullable) {
        zodType = zodType.unwrap();
    }

    // --- Map Core Zod Types to Google Schema Types ---
    if (zodType instanceof ZodString) {
        return {
            type: SchemaType.STRING,
            description: zodType.description,
        };
    } else if (zodType instanceof ZodNumber) {
        const isInt = zodType._def.checks?.some(ch => ch.kind === 'int');
        return {
            type: isInt
                ? SchemaType.INTEGER
                : SchemaType.NUMBER,
            description: zodType.description,
        };
    } else if (zodType instanceof ZodBoolean) {
        return {
            type: SchemaType.BOOLEAN,
            description: zodType.description,
        };
    } else if (zodType instanceof ZodEnum) { // Handles z.enum()
        return {
            type: SchemaType.STRING,
            description: zodType.description,
            enum: [...zodType._def.values], // Copy enum values
            format: "enum"
        };
    } else if (zodType instanceof ZodArray) {
        return {
            type: SchemaType.ARRAY,
            description: zodType.description,
            items: zodTypeToGoogleSchema(zodType._def.type)
        };
    } else if (zodType instanceof ZodObject) {
        const objectSchema: Schema = {
            type: SchemaType.OBJECT,
            description: zodType.description,
            properties: {}, // Initialize properties for the nested object
        };
        const shape = zodType.shape; // Use .shape for ZodObject
        const nestedRequired: string[] = [];
        for (const key in shape) {
            // Recursively convert nested properties
            objectSchema.properties[key] = zodTypeToGoogleSchema(shape[key]);
            // Check if nested property is required (not optional)
             if (!(shape[key] instanceof ZodOptional || shape[key]._def?.typeName === 'ZodOptional')) {
                 // Crude check, might need refinement based on Zod version
                 // nestedRequired.push(key); // Google Schema doesn't have per-nested-property 'required'. Handled at the parent level.
             }
        }
         // Note: Standard JSON Schema for Gemini doesn't typically have 'required' within nested object definitions,
         // it's usually only at the top level of the parameters.
         // definition.required = nestedRequired; // Usually not needed/supported here
         return objectSchema;
    } else {
        return {
            type: SchemaType.STRING,
            description: zodType.description,
        };
    }
}

async function main() {
    const mcpClient = new MCPClient();
    try {
        for (const path of SERVER_PATHS) {
            await mcpClient.connectToServer(path);
        }
        await mcpClient.chatLoop();
    } finally {
       await mcpClient.cleanup();
        process.exit(0);
    }
}
  
main();
