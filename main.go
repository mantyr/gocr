package main

import (
	"./gocr_math"
	. "./neural"
	"fmt"
	"math"
)

var zero = []float64{
	0, 1, 1, 0,
	1, 0, 0, 1,
	1, 0, 0, 1,
	1, 0, 0, 1,
	0, 1, 1, 0,
}
var zero1 = []float64{
	1, 1, 1, 1,
	1, 0, 0, 1,
	1, 0, 0, 1,
	1, 0, 0, 1,
	1, 1, 1, 1,
}

var one = []float64{
	0, 0, 1, 0,
	0, 0, 1, 0,
	0, 0, 1, 0,
	0, 0, 1, 0,
	0, 0, 1, 0,
}

var two = []float64{
	0, 1, 1, 0,
	1, 0, 0, 1,
	0, 0, 1, 0,
	0, 1, 0, 0,
	1, 1, 1, 1,
}

var three = []float64{
	1, 1, 1, 1,
	0, 0, 0, 1,
	0, 1, 1, 1,
	0, 0, 0, 1,
	1, 1, 1, 1,
}

var dataset = [][][]float64{
	{zero1, {0, 0}},
	{zero, {0, 0}},
	{one, {0, 1}},
	{two, {1, 0}},
	{three, {1, 1}},
}

var someData = []struct {
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
	network.Train(dataset)
	fmt.Println("\n\nTrained the data set")

	for _, sumStruct := range someData {

		outputs := network.Process(sumStruct.matrix)

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
		fmt.Printf("Expected %v, got %v (binary: %v\n", sumStruct.decimal, decimal, binary)
	}
}
