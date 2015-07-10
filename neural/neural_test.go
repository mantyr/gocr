package neural

import (
	"testing"
)

type MockProcessor struct {
	retvals []float64
}

func Build_MockProcessor(vals []float64) MockProcessor {
	return MockProcessor{retvals: vals}
}

func (m *MockProcessor) Process(inputs []float64) []float64 {
	return m.retvals
}

func testEq(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

var sizes = []int{1, 2, 5, 9, 13, 20, 21}

func TestBuildNeuron(t *testing.T) {
	var n *Neuron
	for _, size := range sizes {
		n = Build_Neuron(size)

		if len(n.Weights) != size {
			t.Errorf("Expected %v weights, got: %v ", size, len(n.Weights))
		}

		for _, w := range n.Weights {
			if w == 0 {
				t.Errorf("Expected non-zero weight, got zero")
			}
		}

		if n.Bias == 0 {
			t.Errorf("Neuron should have bias")
		}
	}
}

func TestNeuronProcess(t *testing.T) {
	vals := []float64{5, 10}

	n := Build_Neuron(2)
	n.Weights[0] = 2
	n.Weights[1] = 2
	n.Bias = 10

	n.Process(vals)

	if n.lastOutput == 0 {
		t.Errorf("Exp non-zero for lastOutput, got 0")
	}

	if !testEq(n.lastInputs, vals) {
		t.Errorf("Exp %v, got %v", n.lastInputs, vals)
	}
}

func TestBuildLayer(t *testing.T) {
	l := Build_Layer(2, 3)

	if len(l.Neurons) != 2 {
		t.Errorf("Exp 2 neurons, got: %v", len(l.Neurons))
	}

	if len(l.Neurons[0].Weights) != 3 {
		t.Errorf("Exp 2 weights for each neuron, got: %v", len(l.Neurons[0].Weights))
	}
}

func TestLayerProcess(t *testing.T) {
	inputs := []float64{1, 0}
	l := Build_Layer(2, 3)
	retvals := l.Process(inputs)

	for _, val := range retvals {
		if val == 0 {
			t.Errorf("Expect a value greater than 0, got 0")
		}
	}
}

func TestBuildNetwork(t *testing.T) {
	n := Build_Network()

	if (cap(n.Layers)) < 10 {
		t.Errorf("Expected at least %v layers, got %v", 10, cap(n.Layers))
	}
}

func TestNetworkProcess(t *testing.T) {

	inputs := []float64{1, 0}

	n := Build_Network()

	retvals := n.Process(inputs)
	if len(retvals) == 0 {
		//t.Errorf("Expect a value greater than 0, got 0")
	}

	for _, val := range retvals {
		if val == 0 {
			t.Errorf("Expect a value greater than 0, got 0")
		}
	}
}

func TestNetworkAddLayer(t *testing.T) {
	n := Build_Network()

	for i := 0; i < 10; i++ {
		n.AddLayer(0, 1)
		if len(n.Layers) != i+1 {
			t.Errorf("Expect %v, got %v", len(n.Layers), i)
		}
	}
}
