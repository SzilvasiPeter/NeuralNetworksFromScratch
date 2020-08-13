using System;
using System.Collections.Generic;

namespace NeuralNetworksFromScratch
{
    public class Matrix
    {
        public List<List<double>> Row { get; }
        public List<List<double>> Column { get; }

        public int ColumnNumber { get; set; }

        public Matrix()
        {
            Row = new List<List<double>>();
            Column = new List<List<double>>();
        }

        public void AddRow(List<double> row)
        {
            Row.Add(row);
            ColumnNumber++;
        }
    }
    class Program
    {
        private static readonly List<double> Inputs = new List<double>() { 1.0, 2.0, 3.0, 2.50 };
        
        private static readonly Matrix Weights = new Matrix();

        private static readonly List<double> Biases = new List<double>(){ 2.0, 3.0, 0.5};

        public static void Main(string[] args)
        {
            Weights.AddRow(new List<double>(){0.20, 0.80, -0.50, 1.0});
            Weights.AddRow(new List<double>(){0.50, -0.910, 0.260, -0.50});
            Weights.AddRow(new List<double>(){-0.260, -0.270, 0.170, 0.870});

            List<double> outputs = DotProduct(Inputs, Weights, Biases);
            
            foreach (var output in outputs)
            {
                Console.WriteLine(output);
            }
        }

        public static List<double> DotProduct(List<double> inputVector, Matrix leftMatrix, List<double> Biases)
        {
            List<double> outputVector = new List<double>();
            int columnNumber = leftMatrix.ColumnNumber;
            for (int i = 0; i < columnNumber; i++)
            {
                double outputElement = 0;
                for (int j = 0; j < leftMatrix.Row[i].Count; j++)
                {
                    outputElement += leftMatrix.Row[i][j] * inputVector[j];
                }

                outputElement += Biases[i];
                outputVector.Add(outputElement);
            }

            return outputVector;
        }
    }
}