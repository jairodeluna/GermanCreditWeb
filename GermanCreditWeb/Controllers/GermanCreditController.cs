using CsvHelper;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;

namespace GermanCreditWeb.Controllers
{
    public class GermanCreditController : Controller
    {
        private MLContext _mlContext;
        private PredictionEngine<LoanApplicationData, CreditPrediction> _predictionEngine;
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult ImportData()
        {
            var file = Request.Form.Files.FirstOrDefault();
            
            
            if (file != null && file.Length > 0)
            {
                // Initialize MLContext
                _mlContext = new MLContext();

                var uploadsFolder = Path.Combine(Directory.GetCurrentDirectory(), "Uploads");
                if (!Directory.Exists(uploadsFolder))
                {
                    Directory.CreateDirectory(uploadsFolder);
                }
                var filePath = Path.Combine(uploadsFolder, file.FileName);
                using (var stream = new FileStream(filePath, FileMode.Create))
                {
                    file.CopyTo(stream);
                }
               
                // Load and train the model
                var data = _mlContext.Data.LoadFromTextFile<LoanApplicationData>(filePath, hasHeader: true, separatorChar: ',');
                var dataProcessingPipeline = _mlContext.Transforms.Concatenate("Features", "CreditAmount", "Duration")
                    .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", "IsGoodCredit"))
                    .Append(_mlContext.Transforms.NormalizeMinMax("Features"));

                var trainer = _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression();
                var trainingPipeline = dataProcessingPipeline.Append(trainer);
                var trainedModel = trainingPipeline.Fit(data);

                // Create prediction engine
                _predictionEngine = _mlContext.Model.CreatePredictionEngine<LoanApplicationData, CreditPrediction>(trainedModel);
            }
            return View("Index");
        }



        [HttpPost]
        public IActionResult PredictCreditRating(LoanApplicationData loanApplication)
        {
            // Predict credit rating
            var prediction = _predictionEngine.Predict(loanApplication);

            // Display prediction result
            ViewBag.PredictionResult = prediction.IsGoodCredit ? "Good" : "Bad";

            return View("Index");
        }
    }

    public class LoanApplicationData
    {
        [LoadColumn(6)]
        public float CreditAmount { get; set; }

        [LoadColumn(3)]
        public float Duration { get; set; }

        [LoadColumn(22)]
        public uint IsGoodCredit { get; set; } 
    }

    public class CreditPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGoodCredit { get; set; }
    }
}

