package main

import (
  "math/rand"
  "math"
)

// Random weight between -0.2 and 0.2
func Rand() float64 {
  return rand.Float64() * 0.4 - 0.2
}

// Mean squared error
func MSE(errors []float64) float64 {
  sum := 0.0

  for _, i := range errors {
    sum = sum + float64(i) * float64(i)
  }
  return sum / float64(len(errors))
}


func Sum(array []float64) float64 {
  sum := 0.0

  for _, i := range array {
    sum = sum + float64(i)
  }
  return sum
}

func Sigmoid(x float64) float64 {
  return 1 / ( 1 + math.Pow(math.E, -x))
}
