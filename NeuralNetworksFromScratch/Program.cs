﻿using System;
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
        
        public static List<double> DotProduct(List<double> inputVector, Matrix weightsMatrix, List<double> biases)
        {
            List<double> outputVector = new List<double>();
            int columnNumber = weightsMatrix.ColumnNumber;
            for (int i = 0; i < columnNumber; i++)
            {
                double outputElement = 0;
                for (int j = 0; j < weightsMatrix.Row[i].Count; j++)
                {
                    outputElement += weightsMatrix.Row[i][j] * inputVector[j];
                }

                outputElement += biases[i];
                outputVector.Add(outputElement);
            }

            return outputVector;
        }
    }
    class Program
    {
        private static readonly List<double> Inputs = new List<double>() { 1.0, 2.0, 3.0, 2.50 };
        
        private static readonly Matrix Weights = new Matrix();

        private static readonly List<double> Biases = new List<double>(){ 2.0, 3.0, 0.5, 0.5};

        public static void Main(string[] args)
        {
            // Weights.AddRow(new List<double>(){0.20, 0.80, -0.50, 1.0});
            // Weights.AddRow(new List<double>(){0.50, -0.910, 0.260, -0.50});
            // Weights.AddRow(new List<double>(){-0.260, -0.270, 0.170, 0.870});
            // Weights.AddRow(new List<double>(){-0.260, -0.270, 0.170, 0.870});
            //
            // List<double> outputs = DotProduct(Inputs, Weights, Biases);
            //
            // foreach (var output in outputs)
            // {
            //     Console.WriteLine(output);
            // }
            
            LayerDense layer1 = new LayerDense(4, 5);
            LayerDense layer2 = new LayerDense(5, 2);
            
            layer1.Forward(Inputs);
            layer2.Forward(layer1.Outputs);

            foreach (var output in layer2.Outputs)
            {
                Console.WriteLine(output);
            }
        }

        public static List<double> DotProduct(List<double> inputVector, Matrix weightsMatrix, List<double> biases)
        {
            List<double> outputVector = new List<double>();
            int columnNumber = weightsMatrix.ColumnNumber;
            for (int i = 0; i < columnNumber; i++)
            {
                double outputElement = 0;
                for (int j = 0; j < weightsMatrix.Row[i].Count; j++)
                {
                    outputElement += weightsMatrix.Row[i][j] * inputVector[j];
                }

                outputElement += biases[i];
                outputVector.Add(outputElement);
            }

            return outputVector;
        }
    }
}