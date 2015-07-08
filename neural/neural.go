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
  Laters []Layer
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
