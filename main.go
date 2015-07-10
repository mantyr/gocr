package main;

import (
  "fmt"
  . "./neural"
  "./gocr_math"
)


var zero = []float64{
  0, 1, 1, 0,
  1, 0, 0, 1,
  1, 0, 0, 1,
  1, 0, 0, 1,
  0, 1, 1, 0,
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
  { zero, { 0, 0 } },
  { one, { 0, 1 } },
  { two, { 1, 0 } },
  { three, { 1, 1 } },
}

var matrix = []float64{
  1, 1, 1, 1,
  1, 0, 0, 1,
  1, 0, 0, 1,
  1, 0, 0, 1,
  1, 1, 1, 0,
}

// Convert the outpput to binary and then to deimal

func main() {
  network := Build_Network()
  network.AddLayer(10, 20) // Hidden layer
  network.AddLayer(2, 10) // Output layer, defaults to previous layers ouputs: 10
  network.Train(dataset)
  outputs := network.Process(matrix)
  // ouputs == [ ~1, ~0]

  var binary []float64
  // var decimal []int
  for _, i := range outputs {
    binary = append(binary, gocr_math.Round(i))
  }
  //var decimal = parseInt(binary, 2)
  fmt.Println("Hello ", outputs, binary)
}
