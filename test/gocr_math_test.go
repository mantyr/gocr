package gocr_math 

import "../gocr_math"

import "testing"
func TestRand(t *testing.T) {
  for i:= 0; i < 100; i++ { 
    n := gocr_math.Rand()
    if (n < -0.2) || n > 0.2 {
      t.Error("Expected number between -0.2 and 0.2 got: ", n)
    }
  }
}
