package neural 

import (
  "math/rand"
  "../gocr_math"
)

type Neuron struct {
  Weights []float64
  Bias float64
  lastInputs []float64
  lastOutput float64
  error float64
  delta float64
}

func Build_Neuron(numInputs int) Neuron {
  n := Neuron { 
    Weights: make([]float64, 0, numInputs),
    Bias: rand.Float64(),
  }

  for i := 0; i < cap(n.Weights); i++ {
    n.Weights = append(n.Weights, rand.Float64())
  }

  return n
}

func (n *Neuron) Process(inputs []float64) float64{
  n.lastInputs = inputs

  sum := 0.0
  for i, val := range inputs{
    sum += val * n.Weights[i]
  }
  sum += n.Bias

  n.lastOutput = gocr_math.Sigmoid(sum)
  return n.lastOutput
}

type Layer struct {
  Neurons []Neuron
}

func (l *Layer) Process(inputs []float64) []float64{
  var tmp []float64

  for i, _ := range l.Neurons {
    tmp = append(tmp, l.Neurons[i].Process(inputs))
  }

  return tmp
}

func Build_Layer(numNeurons int, numInputs int) Layer {
  l := Layer{Neurons: make([]Neuron, 0, numNeurons)}

  for i:= 0; i < cap(l.Neurons); i++ {
    l.Neurons = append(l.Neurons, Build_Neuron(numInputs))
  }

  return l
}

type Network struct {
  Layers []Layer
  errorThresshold float64
  trainingIterations int
  earningRate float64
}

func Build_Network() Network {
  return Network { 
    Layers: make([]Layer, 0, 10),
    errorThresshold: 0.00001,
    trainingIterations: 500000,
    earningRate: 0.03,
  }
}

func (n *Network) Process(inputs []float64) []float64 {
  var outputs []float64
  for _, l := range n.Layers {
    outputs = l.Process(inputs)
    inputs = outputs
  }
  return outputs
}

func (n *Network) AddLayer(numNeurons int, numInputs int) {
  if numInputs == 0 {
    previousLayer := n.Layers[len(n.Layers)-1]
    numInputs = len(previousLayer.Neurons)
  }

  n.Layers = append(n.Layers, Build_Layer(numNeurons, numInputs))
}

func (n *Network) train(examples [][]float64)  {
  outputLayer := n.Layers[len(n.Layers)-1]

  for iter := 0; iter < n.trainingIterations; iter++ {
    
    for i, example := range examples  {
      inputs := example[0]
      targets := example[1]

      outputs := n.Process(inputs)

      for j, neuron := range outputLayer.Neurons {
        neuron.error = targets[j] - outputs[j]
        neuron.delta = neuron.lastOutput * (1 - neuron.lastOutput) * neuron.error
      }

      for l_index := len(n.Layers) - 2; l_index >= 0; l_index-- {
        for k, neuron :=  range n.Layers[l_index].Neurons {
/*
//NOTE: Translate this into Go code!

          neuron.error = math.sum(this.layers[l + 1].neurons.
                                  map(function(n) { return n.weights[j] * n.delta }))
          neuron.delta = neuron.lastOutput * (1 - neuron.lastOutput) * neuron.error

          for (var i = 0; i < this.layers[l + 1].neurons.length; i++) {
            var neuron = this.layers[l + 1].neurons[i]

            for (var w = 0; w < neuron.weights.length; w++) {
              neuron.weights[w] += this.learningRate * neuron.lastIntputs[w] * neuron.delta
            }
            neuron.bias += this.learningRate * neuron.delta
          }
        }
      }
    }

    var error = math.mse(outputLayer.neurons.
                         map(function(n) { return n.error }))

    if (it % 10000 === 0) {
      console.log({ iteration: it, mse: error })
    }

    if (error <= this.errorThreshold) {
      return
    }
    
  }
*/
        }
      }

    }
  }

}

