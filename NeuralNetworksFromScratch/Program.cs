using System;
using System.Collections.Generic;

namespace NeuralNetworksFromScratch
{
    class Program
    {
        private static readonly List<double> Inputs = new List<double>()
        {
            1, 2, 3
        };
        private static readonly List<double> Weights = new List<double>()
        {
            0.2, 0.8, -0.5
        };

        private static readonly double Bias = 2;

        public static void Main(string[] args)
        {
            double output = Inputs[0] * Weights[0] + Inputs[1] * Weights[1] + Inputs[2] * Weights[2] + Bias;
            Console.WriteLine(output);
        }
    }
}