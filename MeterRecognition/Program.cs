
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

namespace MeterRecognition
{
    public class MeterRecognition
    {
        private const double SPLIT_SHAPE_WHEN_WIDTH_IS_AT_LEAST_TIMES_HEIGHT = 1.6;

        private const float MIN_BLACK_RATIO_HOR = 0.7f;
        private const float MIN_BLACK_RATIO_VER = 0.5f;

        public const int SAMPLE_WIDTH = 20;
        public const int SAMPLE_HEIGHT = 20;

        public class Shape
        {
            public int MinX { get; set; }
            public int MaxX { get; set; }
            public int MinY { get; set; }
            public int MaxY { get; set; }

        }

        static void BuildSet(string baseFolder, string outputPath, int nrOfSamples)
        {

            const int RANDOM_NOISE = 10;

            Random rnd = new Random();
            using (StreamWriter writer = new StreamWriter(System.IO.File.OpenWrite(outputPath)))
            {
                string headerline = ("digit") + "\t" + string.Join('\t', Enumerable.Range(0, SAMPLE_WIDTH * SAMPLE_HEIGHT).Select(i => "col" + i));
                writer.WriteLine(headerline);

                string[][] filesPerDigit = new string[10][];
                int[] counterPerDigit = new int[10];
                for (int i = 0; i < 10; i++)
                {
                    var files = System.IO.Directory.GetFiles(System.IO.Path.Combine(baseFolder, i + ""));
                    filesPerDigit[i] = files.OrderBy(f => rnd.Next()).ToArray(); // poor man shuffle, which isn't a very good shuffle but whatever
                }

                for (int s = 0; s < nrOfSamples; s++)
                {
                    for (int digit = 0; digit < 10; digit++)
                    {
                        //var digitPath = filesPerDigit[digit][rnd.Next(filesPerDigit[digit].Length)];
                        var digitPath = filesPerDigit[digit][counterPerDigit[digit]];
                        counterPerDigit[digit]++;
                        if (counterPerDigit[digit] >= filesPerDigit[digit].Length) // wrap around
                            counterPerDigit[digit] = 0;

                        bool[,] mask = new bool[SAMPLE_WIDTH, SAMPLE_HEIGHT];
                        using (Bitmap bmp = (Bitmap)Bitmap.FromFile(digitPath))
                        {
                            if (bmp.Width > SAMPLE_WIDTH || bmp.Height > SAMPLE_HEIGHT)
                            {
                                // too large, skip
                            }
                            else
                            {
                                int offsetX = rnd.Next(SAMPLE_WIDTH - bmp.Width);
                                int offsetY = rnd.Next(SAMPLE_HEIGHT - bmp.Height);

                                for (int y = 0; y < bmp.Height; y++)
                                {
                                    for (int x = 0; x < bmp.Width; x++)
                                    {
                                        var col = bmp.GetPixel(x, y);
                                        if (col.R == 255 && col.G == 255 && col.B == 255)
                                            mask[x + offsetX, y + offsetY] = true;
                                    }
                                }


                                for (int i = 0; i < RANDOM_NOISE; i++)
                                {
                                    int x = rnd.Next(SAMPLE_WIDTH);
                                    int y = rnd.Next(SAMPLE_HEIGHT);
                                    mask[x, y] = true;
                                }

                                int[] flattenedMask = new int[SAMPLE_WIDTH * SAMPLE_HEIGHT];
                                int idx = 0;
                                for (int y = 0; y < SAMPLE_HEIGHT; y++)
                                {
                                    for (int x = 0; x < SAMPLE_WIDTH; x++)
                                    {
                                        flattenedMask[idx] = mask[x, y] ? 1 : 0;
                                        idx++;
                                    }
                                }
                                string line = (digit + "") + "\t" + string.Join('\t', flattenedMask);
                                writer.WriteLine(line);
                            }
                        }
                    }
                }
            }
        }

        public class ModelInput
        {
            public float Digit { get; set; }

            public float[] Pixels { get; set; }
        }

        static void Train(string tsvPath)
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 0);

            var trainData = mlContext.Data.LoadFromTextFile(path: tsvPath,
                        columns: new[]
                        {
                            new TextLoader.Column("Digit", DataKind.Single, 0),
                            new TextLoader.Column(nameof(ModelInput.Pixels), DataKind.Single, 1, SAMPLE_WIDTH*SAMPLE_HEIGHT),
                        },
                        hasHeader: true,
                        separatorChar: '\t'
                        );

            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Digit").
                    Append(mlContext.Transforms.Concatenate("Features", nameof(ModelInput.Pixels)).AppendCacheCheckpoint(mlContext));

            var trainer = mlContext.MulticlassClassification.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue("Number", "Label"));

            ITransformer trainedModel = trainingPipeline.Fit(trainData);

            mlContext.Model.Save(trainedModel, trainData.Schema, "trainedmodel.zip");
        }


        public class Prediction
        {
            [ColumnName("Score")]
            public float[] Score;
        }



        static void Main(string[] args)
        {

            // base folder where the labelled samples are in
            var baseFolder = args[0];


            var outFolder = System.IO.Path.Combine(baseFolder, "clean");


         //   GenerateCleanAndDigits(baseFolder);
          //   BuildSet(outFolder, "testfile.tsv", 200000);
             Train("testfile.tsv");
            RunPrediction(baseFolder);
         
          //  GeneratePictureFromSet("testfile.tsv", "testfile.png");
         
            //return;

            //LabelAllImagesInFolder(@"D:\Archive-2");

        }

        private static void GeneratePictureFromSet(string setPath, string outputImagePath)
        {
            var lines = System.IO.File.ReadLines(setPath);
            var rowCount = lines.Count() - 1;

            var samplesPerRow =  (int)Math.Ceiling(Math.Sqrt(rowCount));

            lines = System.IO.File.ReadLines(setPath);
            using (var linesEnum = lines.GetEnumerator())
            {
                linesEnum.MoveNext();
                var headerLine = linesEnum.Current;

                var columns = headerLine.Split('\t');
                // assume square and label + index as format
                var sampleWidth = (int)Math.Sqrt(columns.Length - 1);
                var sampleHeight = sampleWidth;

                using (Bitmap output = new Bitmap(samplesPerRow * sampleWidth, samplesPerRow * sampleHeight, PixelFormat.Format8bppIndexed))
                {
                    BitmapData data = output.LockBits(new Rectangle(0, 0, output.Width, output.Height), ImageLockMode.ReadWrite, PixelFormat.Format8bppIndexed);

                    // Copy the bytes from the image into a byte array
                    byte[] bytes = new byte[data.Height * data.Stride];
                    Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);


                    linesEnum.MoveNext();

                    int cur = 0;
                    while (linesEnum.Current != null)
                    {
                        var parts = linesEnum.Current.Split('\t');

                        var offsetX = (cur % samplesPerRow) * sampleWidth;
                        var offsetY = (cur / samplesPerRow) * sampleHeight;

                      
                        int maskIdx = 0;

                        for (int y = 0; y < sampleHeight; y++)
                        {
                            for (int x = 0; x < sampleWidth; x++)
                            {

                                bytes[(offsetY + y) * data.Stride + (offsetX + x)] = (byte)(parts[1 + maskIdx] == "0" ? 0 : 255);
                                //output.SetPixel(offsetX + x, offsetY + y, parts[1 + maskIdx] == "0" ? Color.Black : Color.White);
                                maskIdx++;
                            }
                        }
                        linesEnum.MoveNext();
                        cur++;
                    }
                    // Copy the bytes from the byte array into the image
                    Marshal.Copy(bytes, 0, data.Scan0, bytes.Length);

                    output.UnlockBits(data);
                    output.Save(outputImagePath);
                }
            }

        }

        private static void LabelAllImagesInFolder(string path)
        {
            var mlContext = new MLContext(seed: 0);

            ITransformer trainedModel = mlContext.Model.Load("trainedmodel.zip", out var modelInputSchema);

            var schemaDef = SchemaDefinition.Create(typeof(ModelInput));
            schemaDef[nameof(ModelInput.Pixels)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, SAMPLE_WIDTH * SAMPLE_HEIGHT);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, Prediction>(trainedModel, inputSchemaDefinition: schemaDef);


            VBuffer<float> keys = default(VBuffer<float>);
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            var labelsArray = keys.DenseValues().ToArray();


            foreach (var imgPath in System.IO.Directory.GetFiles(path))
            {
                var filename = System.IO.Path.GetFileName(imgPath);

                CleanResult cleanResult;
                using (Bitmap bmp = (Bitmap)Bitmap.FromFile(imgPath))
                {
                    cleanResult = Clean(bmp);

                }

                if (cleanResult.Shapes.Count <= 8)
                {
                    string predictedString = MakePrediction(predEngine, labelsArray, cleanResult);
                    Console.WriteLine(filename + "-- > " + predictedString);

                    var newPath = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(imgPath), predictedString + System.IO.Path.GetExtension(imgPath));
                    if (System.IO.File.Exists(newPath))
                    {
                        int i = 0;
                        while (System.IO.File.Exists(newPath))
                        {
                            newPath = System.IO.Path.Combine(System.IO.Path.GetDirectoryName(imgPath), predictedString + "_" + i + System.IO.Path.GetExtension(imgPath));
                            i++;
                        }
                    }
                    System.IO.File.Move(imgPath, newPath);

                }
                else
                {

                }

            }
        }

        private static void RunPrediction(string baseFolder)
        {
            var mlContext = new MLContext(seed: 0);

            ITransformer trainedModel = mlContext.Model.Load("trainedmodel.zip", out var modelInputSchema);

            var schemaDef = SchemaDefinition.Create(typeof(ModelInput));
            schemaDef[nameof(ModelInput.Pixels)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, SAMPLE_WIDTH * SAMPLE_HEIGHT);

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, Prediction>(trainedModel, inputSchemaDefinition: schemaDef);


            VBuffer<float> keys = default(VBuffer<float>);
            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
            var labelsArray = keys.DenseValues().ToArray();


            foreach (var imgPath in System.IO.Directory.GetFiles(baseFolder))
            {
                var filename = System.IO.Path.GetFileName(imgPath);
                using (Bitmap bmp = (Bitmap)Bitmap.FromFile(imgPath))
                {
                    var cleanResult = Clean(bmp);
                    if (cleanResult.Shapes.Count <= 8)
                    {
                        string predictedString = MakePrediction(predEngine, labelsArray, cleanResult);
                        Console.WriteLine(filename + " --> " + predictedString);
                    }
                    else
                    {

                    }

                }
            }
        }

        public static string MakePrediction(PredictionEngine<ModelInput, Prediction> predEngine, float[] labelsArray, CleanResult cleanResult)
        {
            string predictedString = "";
            var shapes = cleanResult.Shapes.OrderBy(s => s.MinX).ToArray();
            foreach (var shape in shapes)
            {

                int shapeWidth = shape.MaxX - shape.MinX + 1;
                int shapeHeight = shape.MaxY - shape.MinY + 1;

                if (shapeWidth > SAMPLE_WIDTH || shapeHeight > SAMPLE_HEIGHT) continue;

                int offsetX = (SAMPLE_WIDTH - shapeWidth) / 2;
                int offsetY = (SAMPLE_HEIGHT - shapeHeight) / 2;

                bool[,] mask = new bool[SAMPLE_WIDTH, SAMPLE_HEIGHT];

                for (int y = shape.MinY; y <= shape.MaxY; y++)
                {
                    for (int x = shape.MinX; x <= shape.MaxX; x++)
                    {
                        var col = cleanResult.Result.GetPixel(x, y);
                        if (col.R == 255 && col.G == 255 && col.B == 255)
                            mask[x - shape.MinX + offsetX, y - shape.MinY + offsetY] = true;
                    }
                }

                float[] flattenedMask = new float[SAMPLE_WIDTH * SAMPLE_HEIGHT];
                int idx = 0;
                for (int y = 0; y < SAMPLE_HEIGHT; y++)
                {
                    for (int x = 0; x < SAMPLE_WIDTH; x++)
                    {
                        flattenedMask[idx] = mask[x, y] ? 1 : 0;
                        idx++;
                    }
                }

                var sample = new ModelInput()
                {
                    Pixels = flattenedMask
                };


                var result = predEngine.Predict(sample);

                int maxIdx = -1;
                float maxScore = float.MinValue;
                for (int i = 0; i < result.Score.Length; i++)
                {
                    if (maxScore < result.Score[i])
                    {
                        maxScore = result.Score[i];
                        maxIdx = i;
                    }
                }

                var predictedLabel = labelsArray[maxIdx];

                predictedString += predictedLabel + "";
            }

            return predictedString;
        }

        private static void GenerateCleanAndDigits(string baseFolder)
        {
            var outFolder = System.IO.Path.Combine(baseFolder, "clean");
            if (!System.IO.Directory.Exists(outFolder))
                System.IO.Directory.CreateDirectory(outFolder);

            for (int i = 0; i < 10; i++)
            {
                var digitFolder = System.IO.Path.Combine(baseFolder, "clean", i + "");
                if (!System.IO.Directory.Exists(digitFolder))
                    System.IO.Directory.CreateDirectory(digitFolder);
            }

            //using (var engine = new TesseractEngine(@".", "eng", EngineMode.Default))
            //{
            //    engine.SetVariable("tessedit_char_whitelist", "0123456789");


            int[] digitCounter = new int[10];
            HashSet<string>[] masks = new HashSet<string>[10];
            for (int i = 0; i < 10; i++)
                masks[i] = new HashSet<string>();

            foreach (var imgPath in System.IO.Directory.GetFiles(baseFolder))
            {
                var filename = System.IO.Path.GetFileName(imgPath);
                using (Bitmap bmp = (Bitmap)Bitmap.FromFile(imgPath))
                {
                    var cleanResult = Clean(bmp);


                    var outputPath = System.IO.Path.Combine(outFolder, filename);
                    cleanResult.Result.Save(outputPath);

                    var match = Regex.Match(filename, @"(\d+)(?:.*)");
                    if (match.Success)
                    {
                        var digits = match.Groups[1].Value;
                        var shapes = cleanResult.Shapes.OrderBy(s => s.MinX).ToArray();
                        if (shapes.Length == digits.Length)
                        {
                            for (int i = 0; i < shapes.Length; i++)
                            {
                                var digit = digits[i] - '0';
                                var shape = shapes[i];


                                string mask = "";
                                using (Bitmap shapePixels = new Bitmap(shape.MaxX - shape.MinX + 1, shape.MaxY - shape.MinY + 1))
                                {
                                    for (int y = 0; y < shapePixels.Height; y++)
                                    {
                                        for (int x = 0; x < shapePixels.Width; x++)
                                        {
                                            var col = cleanResult.Result.GetPixel(shape.MinX + x, shape.MinY + y);
                                            if (col.R == 255 && col.G == 255 && col.B == 255)
                                            {
                                                shapePixels.SetPixel(x, y, Color.White);
                                                mask += "1";
                                            }
                                            else
                                            {
                                                shapePixels.SetPixel(x, y, Color.Black);
                                                mask += "0";
                                            }
                                        }
                                    }

                                    // make sure to only save the unique masks
                                    if (!masks[digit].Contains(mask))
                                    {
                                        var digitPath = System.IO.Path.Combine(outFolder, digit + "", digitCounter[digit] + ".png");
                                        shapePixels.Save(digitPath);
                                        digitCounter[digit]++;
                                        masks[digit].Add(mask);
                                    }
                                }
                            }
                        }
                        else
                        {
                            Console.WriteLine("Segmentation failed for " + filename);
                        }
                    }
                    //using (Pix pix = Pix.LoadFromFile(outputPath))
                    //{
                    //    using (var page = engine.Process(pix))
                    //    {

                    //        string result = page.GetText();
                    //        Console.WriteLine(System.IO.Path.GetFileName(imgPath) + " --> " + result.Trim());
                    //    }
                    //}
                }
            }
            //}







            //BitmapData bmpdata = bmp.LockBits(       new Rectangle(0, 0, width, height), System.Drawing.Imaging.ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            //bmpdata.
        }

        public class CleanResult
        {
            public Bitmap Result { get; set; }
            public List<Shape> Shapes { get; set; }
            //public List<int> DigitOffsets { get; internal set; }
        }

        public static CleanResult Clean(Bitmap bmp)
        {
            var width = bmp.Width;
            var height = bmp.Height;

            var grayImage = new byte[width, height];

            var mask = new bool[width, height];

            byte max = byte.MinValue;
            byte min = byte.MaxValue;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var color = bmp.GetPixel(x, y);
                    var gray = (color.R + color.G + color.B) / 3;
                    grayImage[x, y] = (byte)gray;

                    if (max < gray) max = (byte)gray;
                    if (min > gray) min = (byte)gray;
                }
            }

            const float MAX_VALUE_BEFORE_SEEN_AS_WHITE = 0.4f;

            // simple thresholding on normalized image
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    var range = (grayImage[x, y] - min) / (float)(max - min);
                    mask[x, y] = range < MAX_VALUE_BEFORE_SEEN_AS_WHITE ? false : true;
                }
            }
            // Dump(width, height, mask);


            int leftBoundary = int.MaxValue;
            int rightBoundary = int.MaxValue;
            int topBoundary = int.MaxValue;
            int bottomBoundary = int.MaxValue;

            // keep only rows that have more than 70% black pixels, and only at the bottom/top
            // top -> bottom
            for (int y = 0; y < height; y++)
            {
                topBoundary = y;
                int nrBlack = 0;
                for (int x = 0; x < width; x++)
                    if (!mask[x, y]) nrBlack++;

                float ratio = nrBlack / (float)width;
                if (ratio >= MIN_BLACK_RATIO_HOR)
                {
                    // encountered a row that's less than the required amount of pixels, stop blanking out
                    break;
                }
            }

            // bottom -> top
            for (int y = height - 1; y >= 0; y--)
            {
                bottomBoundary = y;
                int nrBlack = 0;
                for (int x = 0; x < width; x++)
                    if (!mask[x, y]) nrBlack++;

                float ratio = nrBlack / (float)width;
                if (ratio >= MIN_BLACK_RATIO_HOR)
                {
                    // encountered a row that's less than the required amount of pixels, stop blanking out
                    break;
                }
            }

            // left -> right
            for (int x = 0; x < width; x++)
            {
                leftBoundary = x;
                int nrBlack = 0;
                for (int y = 0; y < height; y++)
                    if (!mask[x, y]) nrBlack++;

                float ratio = nrBlack / (float)height;
                if (ratio >= MIN_BLACK_RATIO_VER)
                {
                    // encountered a row that's less than the required amount of pixels, stop blanking out
                    break;
                }
            }

            // right -> left
            for (int x = width - 1; x >= 0; x--)
            {
                rightBoundary = x;
                int nrBlack = 0;
                for (int y = 0; y < height; y++)
                    if (!mask[x, y]) nrBlack++;

                float ratio = nrBlack / (float)height;
                if (ratio >= MIN_BLACK_RATIO_VER)
                {
                    // encountered a row that's less than the required amount of pixels, stop blanking out
                    break;
                }
            }



            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if ((x < leftBoundary || x > rightBoundary) || (y < topBoundary || y > bottomBoundary))
                        mask[x, y] = false;
                }
            }
            //Dump(width, height, mask);

            int[] whiteOnY = new int[width];
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    if (mask[x, y])
                        whiteOnY[x]++;
                }
            }

            var boundsMask = (bool[,])mask.Clone();

            boundsMask = Morphology(boundsMask, width, height, erode: false);
            boundsMask = Morphology(boundsMask, width, height, erode: false);
            // Dump(width, height, boundsMask);

            boundsMask = Morphology(boundsMask, width, height, erode: true);
            boundsMask = Morphology(boundsMask, width, height, erode: true);
            //  Dump(width, height, boundsMask);

            //const float REQUIRED_CONSECUTIVE_BLACK_BEFORE_SPLIT_POINT_PERC = 0.02f;
            //const float MAX_WHITE_PERC_BEFORE_SEEN_AS_WHITE_FOR_SPLIT_POINT = 0.15f;
            //List<int> xSplitPoints = new List<int>();
            //int curSplitPoint = leftBoundary;
            //int consecutiveBlackCount = 0;
            //for (int x = leftBoundary; x < width; x++)
            //{
            //    if (whiteOnY[x] / (float)height < MAX_WHITE_PERC_BEFORE_SEEN_AS_WHITE_FOR_SPLIT_POINT)
            //        consecutiveBlackCount++;
            //    else
            //    {
            //        if(consecutiveBlackCount/(float)width >= REQUIRED_CONSECUTIVE_BLACK_BEFORE_SPLIT_POINT_PERC)
            //            xSplitPoints.Add(curSplitPoint);

            //        curSplitPoint = x;
            //        consecutiveBlackCount = 0;
            //    }
            //}



            //Dump(width, height, boundsMask);
            List<Shape> shapes = new List<Shape>();
            bool[,] visited = new bool[width, height];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // search for shapes on the half height
                    if (boundsMask[x, y] && !visited[x, y])
                    {
                        // flood fill on current shape
                        Queue<Point> points = new Queue<Point>();
                        points.Enqueue(new Point(x, y));

                        Shape shape = new Shape()
                        {
                            //Points = new List<Point>(),
                            MaxX = int.MinValue,
                            MaxY = int.MinValue,
                            MinX = int.MaxValue,
                            MinY = int.MaxValue
                        };


                        while (points.Count > 0)
                        {
                            var curPoint = points.Dequeue();
                            if (!visited[curPoint.X, curPoint.Y] && boundsMask[curPoint.X, curPoint.Y])
                            {
                                //shape.Points.Add(curPoint);
                                if (curPoint.X < shape.MinX) shape.MinX = curPoint.X;
                                if (curPoint.Y < shape.MinY) shape.MinY = curPoint.Y;
                                if (curPoint.X > shape.MaxX) shape.MaxX = curPoint.X;
                                if (curPoint.Y > shape.MaxY) shape.MaxY = curPoint.Y;


                                visited[curPoint.X, curPoint.Y] = true;

                                // visit neighbours
                                if (curPoint.X - 1 >= 0) points.Enqueue(new Point(curPoint.X - 1, curPoint.Y));
                                if (curPoint.Y - 1 >= 0) points.Enqueue(new Point(curPoint.X, curPoint.Y - 1));
                                if (curPoint.X + 1 < width) points.Enqueue(new Point(curPoint.X + 1, curPoint.Y));
                                if (curPoint.Y + 1 < height) points.Enqueue(new Point(curPoint.X, curPoint.Y + 1));
                            }
                        }

                        int shapeWidth = shape.MaxX - shape.MinX + 1;
                        int shapeHeight = shape.MaxY - shape.MinY + 1;
                        if (shapeHeight >= height / 3) // at least 1/3rd size
                            shapes.Add(shape);
                    }
                }
            }


            for (int i = shapes.Count - 1; i >= 0; i--)
            {
                int shapeWidth = shapes[i].MaxX - shapes[i].MinX + 1;
                int shapeHeight = shapes[i].MaxY - shapes[i].MinY + 1;
                if (shapeWidth >= SPLIT_SHAPE_WHEN_WIDTH_IS_AT_LEAST_TIMES_HEIGHT * shapeHeight)
                {
                    // split the shape in 2 because it was concatenated due to dilation

                    Shape newShape = new Shape()
                    {
                        MinX = shapes[i].MinX + shapeWidth / 2,
                        MaxX = shapes[i].MaxX,
                        MinY = shapes[i].MinY,
                        MaxY = shapes[i].MaxY
                    };
                    shapes[i].MaxX = newShape.MinX;
                    shapes.Add(newShape);
                }
            }

            Bitmap test = new Bitmap(width, height);

            var shapeMask = new bool[width, height];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                    test.SetPixel(x, y, Color.Black);
            }

            foreach (var shape in shapes)
            {
                for (int y = shape.MinY; y <= shape.MaxY; y++)
                {
                    for (int x = shape.MinX; x <= shape.MaxX; x++)
                    {
                        if (x == shape.MinX || x == shape.MaxX || y == shape.MinY || y == shape.MaxY)
                            test.SetPixel(x, y, Color.Red);

                        if (mask[x, y])
                            test.SetPixel(x, y, Color.White);
                    }
                }

            }


            return new CleanResult()
            {
                Result = test,
                //DigitOffsets = xSplitPoints
                Shapes = shapes
            };
        }

        private static void Dump(int width, int height, bool[,] mask)
        {

            Bitmap test = new Bitmap(width, height);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    test.SetPixel(x, y, mask[x, y] ? Color.White : Color.Black);
                }
            }
            test.Save(@"D:\tempoutput.png", System.Drawing.Imaging.ImageFormat.Png);
        }

        private static bool[,] Morphology(bool[,] img, int width, int height, bool erode)
        {

            bool[,] newImg = new bool[width, height];
            // very inefficient but whatever
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int nrBlack = 0;
                    int nrWhite = 0;

                    //if (x - 1 >= 0 && img[x - 1, y]) nrWhite++; else nrBlack++;
                    //if (y - 1 >= 0 && img[x, y - 1]) nrWhite++; else nrBlack++;
                    //if (x + 1 < width && img[x + 1, y]) nrWhite++; else nrBlack++;
                    //if (y + 1 < height && img[x, y + 1]) nrWhite++; else nrBlack++;


                    for (int l = -1; l <= 1; l++)
                    {
                        for (int k = -1; k <= 1; k++)
                        {
                            if ((l == 0 && k == 0)) continue;

                            if (x + k >= 0 && y + l >= 0 && x + k < width && y + l < height)
                            {
                                if (img[x + k, y + l])
                                    nrWhite++;
                                else
                                    nrBlack++;
                            }
                        }
                    }

                    if (erode)
                    {
                        if (nrBlack > 0)
                            newImg[x, y] = false;
                        else
                            newImg[x, y] = img[x, y];
                    }
                    else
                    {
                        if (nrWhite > 0)
                            newImg[x, y] = true;
                        else
                            newImg[x, y] = img[x, y];
                    }
                }
            }

            return newImg;

        }
    }
}
