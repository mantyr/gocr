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

func TestSum(t *testing.T) {
  var tests = []struct{
    nums []float64
    total float64
  }{
    { []float64{1.0,2.0,3.0} , 6.0 },
    { []float64{10.0,2.0,3.0} , 15.0 },
    { []float64{100.0,2.0,3.0} , 105.0 },
    { []float64{1000.0,2.0,3.0} , 1005.0 },
  }

  for _, test := range tests {
    if Sum(test.nums) != test.total {

      t.Errorf("Expected %v, got %v", test.total, Sum(test.nums))
    }
  }
}

func TestRound(t *testing.T) {
  var tests = []struct{
    num float64
    rounded float64
  }{
    { -1.1, -1},
    { -1.6, -2},
    {  3.6, 4},
    { 3.3, 3},
    {},
  }

  for _, test := range tests {
    if Round(test.num) != test.rounded{

      t.Errorf("Expected %v, got %v", test.rounded, Round(test.num))
    }
  }
}
