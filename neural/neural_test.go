package neural

import (
  "testing"
)

var sizes = []int{1,2,5,9,13,20,21}
func TestBuildNeuron(t *testing.T) {
  var n Neuron
  for _, size := range sizes { 
    n = Build_Neuron(size)
    if len(n.Weights) != size {
      t.Errorf("Expected ", size, " Weights, got: ", len(n.Weights))
    }
  }
}
