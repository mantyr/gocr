package gocr_math 

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

func Round(n float64) float64 {
  if n < 0 {
    return math.Ceil(n-0.5)
  }
  return math.Floor(n+0.5)
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
