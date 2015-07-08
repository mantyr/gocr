package main;

import (
  "fmt"
  "./neural"
)

var network Network =  Build_Network()

var matrix = []uint8{
  1,1,1,1,
  1,0,0,1,
  0,0,1,0,
  0,1,0,0,
  1,1,1,0,
}


//var outputs = network.process(matrix)

// ouputs == [ ~1, ~0]
// Convert the outpput to binary and then to deimal

func main() {
  network.AddLayer(10, 20) // Hidden layer
  network.AddLayer(2) // Output layer, defaults to previous layers ouputs: 10
  //  var binary []int
  // var decimal []int
  fmt.Println("Hello %+v", matrix)
}
