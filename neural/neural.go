package neural 

import (
  "math/rand"
  "../gocr_math"
  "fmt"
)

type Processor interface {
  Process([]float64) []float64 
}

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

func Build_Layer(numNeurons int, numInputs int) Layer {
  l := Layer{Neurons: make([]Neuron, 0, numNeurons)}

  for i:= 0; i < cap(l.Neurons); i++ {
    l.Neurons = append(l.Neurons, Build_Neuron(numInputs))
  }

  return l
}

func (l *Layer) Process(inputs []float64) []float64{
  var tmp []float64

  for i, _ := range l.Neurons {
    tmp = append(tmp, l.Neurons[i].Process(inputs))
  }

  return tmp
}

type Network struct {
  Layers []Layer
  errorThreshold float64
  trainingIterations int
  learningRate float64
}

func Build_Network() Network {
  return Network { 
    Layers: make([]Layer, 0, 10),
    errorThreshold: 0.00001,
    trainingIterations: 500000,
    learningRate: 0.03,
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

func (n *Network) Train(examples [][][]float64)  {
  outputLayer := n.Layers[len(n.Layers)-1]

  for iter := 0; iter < n.trainingIterations; iter++ {
    
    for _, example := range examples  {
      inputs := example[0]
      targets := example[1]

      outputs := n.Process(inputs)

      for j, neuron := range outputLayer.Neurons {
        neuron.error = targets[j] - outputs[j]
        neuron.delta = neuron.lastOutput * (1 - neuron.lastOutput) * neuron.error
      }

      for l_index := len(n.Layers) - 2; l_index >= 0; l_index-- {
        for k, neuron :=  range n.Layers[l_index].Neurons {
          var tmpError []float64
          for _, neuron := range n.Layers[l_index+1].Neurons {
            tmpError = append(tmpError, neuron.Weights[k] * neuron.delta)
          }
          neuron.error = gocr_math.Sum(tmpError)
          fmt.Println(tmpError)
          neuron.delta = neuron.lastOutput * (1 - neuron.lastOutput) * neuron.error

          for _, inner_neuron := range n.Layers[l_index+1].Neurons {
            for w := 0; w < len(inner_neuron.Weights); w++  {
              inner_neuron.Weights[w] += n.learningRate * inner_neuron.lastInputs[w] * inner_neuron.delta
            }
            inner_neuron.Bias += n.learningRate * inner_neuron.delta
          }
        }
      }

      var neuronErrors []float64
      for _, neuron := range outputLayer.Neurons {
        neuronErrors = append(neuronErrors, neuron.error)
      }
      mse := gocr_math.MSE(neuronErrors)

      if iter % 10000 ==0 {
        fmt.Println("iteration: ", iter, " mse: ", mse)
      }

      if mse <= n.errorThreshold {
        return
      }
    }
  }
}
