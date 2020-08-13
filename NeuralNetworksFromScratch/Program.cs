using System;
using System.Collections.Generic;

namespace NeuralNetworksFromScratch
{
    class Program
    {
        private static readonly List<double> Inputs = new List<double>() { 1, 2, 3, 2.5 };
        
        private static readonly List<List<double>> Weights = new List<List<double>>()
        {
            new List<double>(){0.2, 0.8, -0.5, 1.0},
            new List<double>(){0.5, -0.91, 0.26, -0.5},
            new List<double>(){-0.26, -0.27, 0.17, 0.87}
        };

        private static readonly List<double> Bias = new List<double>(){ 2, 3, 0.5};

        public static void Main(string[] args)
        {
            List<double> outputs = new List<double>
            {
                Inputs[0] * Weights[0][0] + Inputs[1] * Weights[0][1] + Inputs[2] * Weights[0][2] + Inputs[3] * Weights[0][3] + Bias[0],
                Inputs[0] * Weights[1][0] + Inputs[1] * Weights[1][1] + Inputs[2] * Weights[1][2] + Inputs[3] * Weights[1][3] + Bias[1],
                Inputs[0] * Weights[2][0] + Inputs[1] * Weights[2][1] + Inputs[2] * Weights[2][2] + Inputs[3] * Weights[2][3] + Bias[2]
            };
            foreach (var output in outputs)
            {
                Console.WriteLine(output);
            }
        }
    }
}