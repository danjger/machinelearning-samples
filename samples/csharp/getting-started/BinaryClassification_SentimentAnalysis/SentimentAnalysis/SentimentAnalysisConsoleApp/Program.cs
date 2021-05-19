using System;
using System.IO;
using Microsoft.ML;
using SentimentAnalysisConsoleApp.DataStructures;
using System.Collections.Generic;
using Microsoft.ML.Transforms.Text;
using Common;
using static Microsoft.ML.DataOperationsCatalog;
using CommandLine;
using CommandLine.Text;


namespace SentimentAnalysisConsoleApp
{
    public class CommandLineOptions
    {
        [Option('t', "train", Required = false, HelpText = "Train Model", Default = false)]
        public bool Train { get; set; }
        [Option('i', "interactive", Required = false, HelpText = "Interactive Checker", Default = false)]
        public bool Interactive { get; set; }
        [Option('c', "check", Required = false, HelpText = "Check Value")]
        public string Check { get; set; }
    }
    internal static class Program
    {
        private static readonly string BaseDatasetsRelativePath = @"../../../../Data";
        private static readonly string DataRelativePath = $"{BaseDatasetsRelativePath}/fname_train_shuf.tsv";

        private static readonly string DataPath = GetAbsolutePath(DataRelativePath);

        private static readonly string BaseModelsRelativePath = @"../../../../MLModels";
        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/fnameModel.zip";

        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);
        static void DisplayHelp<T>(ParserResult<T> result)
        {
            var helpText = HelpText.AutoBuild(result, h =>
            {
                h.AdditionalNewLineAfterOption = false;
                h.Heading = "fname detector 1.0.0-beta"; //change header
                h.Copyright = "Copyright (c) 2021 Tropare"; //change copyright text
                return HelpText.DefaultParsingErrorsHandler(result, h);
            }, e => e);
            Console.WriteLine(helpText);
        }
        static void Main(string[] args)
        {
            ParserResult<CommandLineOptions> parseResult = Parser.Default.ParseArguments<CommandLineOptions>(args);
            if (parseResult.Tag == ParserResultType.NotParsed)
            {
                DisplayHelp<CommandLineOptions>(parseResult);
                return;
            }
            parseResult.WithParsed<CommandLineOptions>(options => Run(options));
        }
        static int Run(CommandLineOptions options)
        {

            int trainExit = 0;
            int scoreExit = 0;
            if (options.Train)
            {
                trainExit = RunTrainAndReturnExitCode();
            };
            if (trainExit == 0 && (options.Check != null))
            {
                scoreExit = RunScoreAndReturnExitCode(options.Check);
                return scoreExit;
            }
            if (trainExit == 0 && scoreExit == 0 && options.Interactive)
            {
                return RunInteractive();
            }
            else
            {
                return trainExit + scoreExit;
            }
        }
        static int RunInteractive()
        {
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);
            ITransformer trainedModel;
            using (var file = File.OpenRead(ModelPath))
                trainedModel = mlContext.Model.Load(file, out DataViewSchema schema);
            if (trainedModel == null)
                return -1;
            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);

            Console.WriteLine("Enter value to check:");

            // LOOP through stdin
            var checkValue = Console.ReadLine();
            while (checkValue != "")
            {
                SentimentIssue sampleStatement = new SentimentIssue { Text = checkValue };


                // Score
                var resultprediction = predEngine.Predict(sampleStatement);
                Console.WriteLine($"Text: {sampleStatement.Text} | Tokens: {string.Join(",", resultprediction.OutputTokens)} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "fname" : "not fname")} | Probability of being fname: {resultprediction.Probability} ");
                checkValue = Console.ReadLine();
            }
            return 0;
        }
        static int RunScoreAndReturnExitCode(string check)
        {
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);
            ITransformer trainedModel;
            using (var file = File.OpenRead(ModelPath))
                trainedModel = mlContext.Model.Load(file, out DataViewSchema schema);
            if (trainedModel == null)
                return -1;

            // TRY IT: Make a single test prediction, loading the model from .ZIP file
            SentimentIssue sampleStatement = new SentimentIssue { Text = check };

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);

            // Score
            var resultprediction = predEngine.Predict(sampleStatement);
            // Print the length of the feature vector.
            Console.WriteLine($"Number of Features: {resultprediction.Features.Length}");

            /*             // Print feature values and tokens.
                        Console.Write("Features: ");
                        for (int i = 0; i < 10; i++)
                            Console.Write($"{resultprediction.Features[i]:F4}  ");
             */
            Console.WriteLine("\nTokens: " + string.Join(",", resultprediction
                .OutputTokens));

            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "fname" : "not fname")} | Probability of being fname: {resultprediction.Probability} ");
            Console.WriteLine($"================End of Process.Hit any key to exit==================================");
            Console.ReadLine();
            return 0;
        }
        static int RunTrainAndReturnExitCode()
        {

            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            // STEP 1: Common data loading configuration
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);

            TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);
            IDataView trainingData = trainTestSplit.TrainSet;
            IDataView testData = trainTestSplit.TestSet;

            // STEP 2: Common data process configuration with pipeline data transformations          
            var options = new Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Options()
            {
                OutputTokensColumnName = "OutputTokens",

                CaseMode = Microsoft.ML.Transforms.Text.TextNormalizingEstimator.CaseMode.Lower,
                KeepDiacritics = true,
                KeepNumbers = true,
                KeepPunctuations = true,
                StopWordsRemoverOptions = null,
                // Use ML.NET's built-in stop word remover
                //                StopWordsRemoverOptions = new Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Options()
                //                {
                //                    Language = Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Language.English
                //                },
                WordFeatureExtractor = null,
                //CharFeatureExtractor = null,
                //WordFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options() { NgramLength = 2, UseAllLengths = true },
                CharFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options() { NgramLength = 3, UseAllLengths = false },
            };
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", options: options, inputColumnNames: nameof(SentimentIssue.Text));
            // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);

            // STEP 5: Evaluate the model and show accuracy stats
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

            ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);

            Console.WriteLine("The model is saved to {0}", ModelPath);
            return 0;
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}