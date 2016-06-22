// Package utilities implements a host of helpful miscellaneous functions to the library.
package utilities

import (
	"encoding/csv"
	"errors"
	"fmt"
	"os"
	"sort"
	"strconv"

	"github.com/gonum/matrix/mat64"
)

type sortedIntMap struct {
	m map[int]float64
	s []int
}

func (sm *sortedIntMap) Len() int {
	return len(sm.m)
}

func (sm *sortedIntMap) Less(i, j int) bool {
	return sm.m[sm.s[i]] < sm.m[sm.s[j]]
}

func (sm *sortedIntMap) Swap(i, j int) {
	sm.s[i], sm.s[j] = sm.s[j], sm.s[i]
}

func SortIntMap(m map[int]float64) []int {
	sm := new(sortedIntMap)
	sm.m = m
	sm.s = make([]int, len(m))
	i := 0
	for key := range m {
		sm.s[i] = key
		i++
	}
	sort.Sort(sm)
	return sm.s
}

func FloatsToMatrix(floats []float64) *mat64.Dense {
	return mat64.NewDense(1, len(floats), floats)
}

func VectorToMatrix(vector *mat64.Vector) *mat64.Dense {
	vec := vector.RawVector()
	return mat64.NewDense(1, len(vec.Data), vec.Data)
}

func SaveMatrixDataToCSV(data *mat64.Dense, labels []string, filename string) error {
	if len(filename) < 4 || filename[len(filename)-3:] != "csv" {
		return errors.New(fmt.Sprint("CSV filename required.", filename))
	}

	f, err := os.Create(filename)
	if err != nil {
		return errors.New(fmt.Sprintf("%s could not be created", filename))
	}
	writer := csv.NewWriter(f)
	rawData := data.RawMatrix().Data
	numCols := data.RawMatrix().Cols

	csvRow := []string{}
	labelCounter := 0
	for index, val := range rawData {
		csvRow = append(csvRow, strconv.FormatFloat(val, 'f', 6, 64))
		if index%numCols == (numCols - 1) {
			csvRow = append(csvRow, labels[labelCounter])
			writer.Write(csvRow)
			labelCounter += 1
			csvRow = []string{}
		}
	}
	writer.Flush()
	return nil
}
