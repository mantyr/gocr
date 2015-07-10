# GoCR
Toy OCR written in go using neural networks


### Create Training Data ####

The data that we train with is a 3-dimension array of floats.
Each element is a 2-dimension array of floats, called a __training set__.
The first element of a __training set__ is the 4 x 5 matrix representation of a number like:

```go
		[]float64{
			1, 1, 1, 0,
			0, 0, 0, 1,
			0, 1, 1, 0,
			0, 0, 0, 1,
			1, 1, 1, 0,
		}
```

Can you see the 3?

The second element of a __training set__ the decimal representation of this number

### Building a Network ###

First you build the network. Then you add layers.

Layers accept a number of inputs and outputs: `AddLayer(outputs, inputs)`

Succeeding layers should contain an identical number of inputs as the preceding layers ouputs, like so:

```go
	network := Build_Network()

    // 20 inputs (4 x 5 matrix == 20 inputs), 10 outputs
	network.AddLayer(10, 20) 

    // 10 inputs (identical to preceding layers inputs)
	network.AddLayer(2, 10)  

    // Final output is 2 numbers, a binary representation of the number

    // Train that data!
	network.Train(trainingData)
```

### Process A Matrix  ###

```go
  var testData = []float64{
			1, 1, 1, 1,
			0, 0, 0, 1,
			0, 0, 1, 0,
			1, 1, 0, 0,
			1, 1, 1, 1,
		}
  ouputs := network.Process(testData)
  // [ float close to one, float close to zero ] -> [1, 0] -> 10b -> 2
```

### Driver ###

Here is the driver:

```go
package main

import (
	"./gocr_math"
	. "./neural"
	"fmt"
	"math"
)

// matrix of number, binary representation of number
var trainingData = [][][]float64{
	{
		[]float64{
			0, 1, 1, 0,
			1, 0, 0, 1,
			1, 0, 0, 1,
			1, 0, 0, 1,
			0, 1, 1, 0,
		}, {0, 0}},
	{
		[]float64{
			1, 1, 1, 1,
			1, 0, 0, 1,
			1, 0, 0, 1,
			1, 0, 0, 1,
			1, 1, 1, 1,
		}, {0, 0}},

	{
		[]float64{
			0, 0, 1, 0,
			0, 0, 1, 0,
			0, 0, 1, 0,
			0, 0, 1, 0,
			0, 0, 1, 0,
		}, {0, 0}},
	{
		[]float64{
			0, 1, 1, 0,
			1, 0, 0, 1,
			0, 0, 1, 0,
			0, 1, 0, 0,
			1, 1, 1, 1,
		}, {0, 0}},

	{
		[]float64{
			1, 1, 1, 1,
			0, 0, 0, 1,
			0, 1, 1, 1,
			0, 0, 0, 1,
			1, 1, 1, 1,
		}, {0, 0}},
}

var testData = []struct {
	matrix  []float64
	decimal int
}{
	{
		// Zero
		[]float64{
			1, 1, 1, 1,
			1, 0, 0, 1,
			1, 0, 0, 1,
			1, 0, 0, 1,
			1, 1, 1, 0,
		}, 0},
	// Zero
	{
		[]float64{
			0, 1, 1, 0,
			1, 0, 0, 1,
			1, 0, 0, 1,
			1, 0, 0, 1,
			0, 1, 1, 0,
		}, 0},
	// Zero
	{
		[]float64{
			0, 1, 1, 1,
			1, 0, 0, 1,
			1, 0, 0, 1,
			1, 0, 0, 1,
			1, 1, 1, 0,
		}, 0},
	{
		// One
		[]float64{
			0, 0, 1, 0,
			0, 0, 1, 0,
			0, 0, 1, 0,
			0, 0, 1, 0,
			0, 0, 1, 0,
		}, 1},
	// Three
	{
		[]float64{
			0, 1, 1, 0,
			0, 0, 0, 1,
			0, 1, 1, 0,
			0, 0, 0, 1,
			0, 1, 1, 0,
		}, 3},
	// Two
	{
		[]float64{
			1, 1, 1, 1,
			0, 0, 0, 1,
			0, 0, 1, 0,
			1, 1, 0, 0,
			1, 1, 1, 1,
		}, 2},
}

// Convert the outpput to binary and then to deimal

func main() {
	network := Build_Network()
	network.AddLayer(10, 20) // Hidden layer
	network.AddLayer(2, 10)  // Output layer, defaults to previous layers ouputs: 10
	network.Train(trainingData)
	fmt.Println("\n\nTrained the data set")

	for _, data := range testData {

		outputs := network.Process(data.matrix)

		var binary []float64
		for _, i := range outputs {
			binary = append(binary, gocr_math.Round(i))
		}

		// Convert to decimal, walk backwards
		var decimal float64
		maxPowers := len(binary) - 1
		for i, v := range binary {
			decimal += math.Pow(2, float64(maxPowers-i)) * v
		}

		//var decimal = parseInt(binary, 2)
		fmt.Printf("Expected %v, got %v (binary: %v\n", data.decimal, decimal, binary)
	}
}
```
