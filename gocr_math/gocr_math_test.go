package gocr_math 

import 
(
   "testing"
  "fmt"
)

func TestRand(t *testing.T) {
  for i:= 0; i < 100; i++ { 
    n := Rand()
    if (n < -0.2) || n > 0.2 {
      fmt.Println(n)
      t.Error("Expected number between -0.2 and 0.2 got: ", n)
    }
  }
}
