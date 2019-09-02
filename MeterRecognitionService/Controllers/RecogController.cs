using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using static MeterRecognition.MeterRecognition;

namespace MeterRecognitionService.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class RecogController : ControllerBase
    {
        private readonly PredictionEngine<ModelInput, Prediction> predEngine;
        private readonly float[] labelsArray;

        public RecogController()
        {
            var mlContext = new MLContext(seed: 0);



            ITransformer trainedModel = mlContext.Model.Load("trainedmodel.zip", out var modelInputSchema);

            var schemaDef = SchemaDefinition.Create(typeof(ModelInput));
            schemaDef[nameof(ModelInput.Pixels)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, SAMPLE_WIDTH * SAMPLE_HEIGHT);

            // Create prediction engine related to the loaded trained model
            predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, Prediction>(trainedModel, inputSchemaDefinition: schemaDef);

            VBuffer<float> keys = default(VBuffer<float>);
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            labelsArray = keys.DenseValues().ToArray();


        }

        [HttpPut("clean")]
        public IActionResult PutClean()
        {

            // Train("testfile.tsv");

            //BuildSet(outFolder, "testfile.tsv", 10000);

            using (Bitmap bmp = (Bitmap)Bitmap.FromStream(Request.Body))
            {
                var cleanResult = MeterRecognition.MeterRecognition.Clean(bmp);

                byte[] bytes;
                using (var ms = new MemoryStream())
                {
                    cleanResult.Result.Save(ms, ImageFormat.Png);
                    bytes = ms.ToArray();
                }
                return File(bytes, "image/png");
            }



        }

        // PUT api/recog/
        [HttpPut]
        public string Put()
        {

            // Train("testfile.tsv");

            //BuildSet(outFolder, "testfile.tsv", 10000);
            
                using (Bitmap bmp = (Bitmap)Bitmap.FromStream(Request.Body))
                {
                    var cleanResult = MeterRecognition.MeterRecognition.Clean(bmp);

                    string predictedString = MeterRecognition.MeterRecognition.MakePrediction(predEngine, labelsArray, cleanResult);
                    return predictedString;
                }
            
          

        }

    }
}
