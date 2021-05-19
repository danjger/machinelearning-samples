
using Microsoft.ML.Data;

namespace SentimentAnalysisConsoleApp.DataStructures
{
    public class SentimentIssue
    {
        [LoadColumn(0)]
        public string Text { get; set; }
        [LoadColumn(1)]
        public bool Label { get; set; }
    }
    
}
