using System;
using System.Collections.Generic;

namespace NeuralNetworksFromScratch
{
    public class LayerDense
    {
        public Matrix Weights { get; set; }
        public List<double> Biases { get; set; }
        public List<double> Outputs { get; set; }

        public LayerDense(int numberOfInputs, int numberOfNeurons)
        {
            Weights = new Matrix();
            Biases = new List<double>();
            for (int i = 0; i < numberOfNeurons; i++)
            {
                List<double> generatedList = new List<double>();
                for (int j = 0; j < numberOfInputs; j++)
                {
                    generatedList.Add(_randomDouble.NextDouble());
                }
                Weights.AddRow(generatedList);
                Biases.Add(0);
            }
        }

        public void Forward(List<double> inputs)
        {
            Outputs = Matrix.DotProduct(inputs, Weights, Biases);
        }
        
        private readonly Random _randomDouble = new Random(0);
    }
}