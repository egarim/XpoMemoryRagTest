using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;

using Microsoft.Extensions.DependencyInjection;

using Microsoft.SemanticKernel.ChatCompletion;

using Microsoft.SemanticKernel.Plugins.Memory;
using System.Diagnostics;
using Microsoft.SemanticKernel.Connectors.Xpo;

#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0003
#pragma warning disable SKEXP0010
#pragma warning disable SKEXP0011
#pragma warning disable SKEXP0050
#pragma warning disable SKEXP0052

namespace XpoMemoryRagTest
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public async Task Test1()
        {

            var question = "What is Bruno's favourite super hero?";
            Debug.WriteLine($"This program will answer the following question: {question}");
            Debug.WriteLine("1st approach will be to ask the question directly to the Phi-3 model.");
            Debug.WriteLine("2nd approach will be to add facts to a semantic memory and ask the question again");
            Debug.WriteLine("");

            var key=   Environment.GetEnvironmentVariable("OpenAiTestKey");

            // Create a chat completion service
            var builder = Kernel.CreateBuilder();
            builder.AddOpenAIChatCompletion(
                modelId: "gpt-4o",
                apiKey: key);
            builder.AddLocalTextEmbeddingGeneration();
            Kernel kernel = builder.Build();

            Debug.WriteLine($"Phi-3 response (no memory).");
            var response = kernel.InvokePromptStreamingAsync(question);
            await foreach (var result in response)
            {
                Debug.Write(result);
            }

            // separator
            Debug.WriteLine("");
            Debug.WriteLine("==============");
            Debug.WriteLine("");

            // get the embeddings generator service
            var embeddingGenerator = kernel.Services.GetRequiredService<ITextEmbeddingGenerationService>();
            //var memory = new SemanticTextMemory(new VolatileMemoryStore(), embeddingGenerator);
            //var  XpoMemoryStore.ConnectAsync("")

            var Store = new VolatileMemoryStore();
            var memory = new SemanticTextMemory(Store, embeddingGenerator);

            // add facts to the collection
            const string MemoryCollectionName = "fanFacts";

            await memory.SaveInformationAsync(MemoryCollectionName, id: "info1", text: "Gisela's favourite super hero is Batman");
            await memory.SaveInformationAsync(MemoryCollectionName, id: "info2", text: "The last super hero movie watched by Gisela was Guardians of the Galaxy Vol 3");
            await memory.SaveInformationAsync(MemoryCollectionName, id: "info3", text: "Bruno's favourite super hero is Invincible");
            await memory.SaveInformationAsync(MemoryCollectionName, id: "info4", text: "The last super hero movie watched by Bruno was Aquaman II");
            await memory.SaveInformationAsync(MemoryCollectionName, id: "info5", text: "Bruno don't like the super hero movie: Eternals");

            TextMemoryPlugin memoryPlugin = new(memory);

            // Import the text memory plugin into the Kernel.
            kernel.ImportPluginFromObject(memoryPlugin);

            OpenAIPromptExecutionSettings settings = new()
            {
                ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions,
            };


            var prompt = @"
            Question: {{$input}}
            Answer the question using the memory content: {{Recall}}";

            var arguments = new KernelArguments(settings)
            {
                { "input", question },
                { "collection", MemoryCollectionName }
            };

            Debug.WriteLine($"Phi-3 response (using semantic memory).");

            response = kernel.InvokePromptStreamingAsync(prompt, arguments);
            await foreach (var result in response)
            {
                Debug.Write(result);
            }

            Debug.WriteLine($"");
            Debug.WriteLine("Hello, World!");
       
            Assert.Pass();
        }
    }
}