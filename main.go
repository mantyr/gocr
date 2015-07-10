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
	network := NewNetwork()
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
