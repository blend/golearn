package text

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/gonum/matrix/mat64"
)

type TfidfVectorizer struct {
	vocabulary []string
	vocabCount map[string]int
	idf        *mat64.Dense
	size       int
}

func NewTfidfVectorizer(vocabulary []string) *TfidfVectorizer {
	sort.Strings(vocabulary)
	vocabCount := make(map[string]int)
	for _, str := range vocabulary {
		vocabCount[str] = 0
	}

	return &TfidfVectorizer{
		vocabulary,
		vocabCount,
		nil,
		len(vocabulary),
	}
}

func (tfidf *TfidfVectorizer) Fit(data []string) error {
	if len(data) == 0 {
		return errors.New(fmt.Sprint("Cannot have 0 vocab words"))
	}

	for _, str := range tfidf.vocabulary {
		tfidf.vocabCount[str] = 0
	}

	for _, textData := range data {
		for _, str := range strings.Split(textData, " ") {
			_, present := tfidf.vocabCount[str]
			if present {
				tfidf.vocabCount[str] += 1
			}
		}
	}
	idfArray := make([]float64, tfidf.size)

	for index, str := range tfidf.vocabulary {
		idfValue := math.Log(float64(tfidf.size) / float64(tfidf.vocabCount[str]))
		idfArray[index] = idfValue
	}

	tfidf.idf = mat64.NewDense(tfidf.size, 1, idfArray)

	return nil
}

func (tfidf *TfidfVectorizer) calculateTF(str string) []float64 {
	tfArray := make([]float64, tfidf.size)
	strMap := make(map[string]int)
	for _, word := range strings.Split(str, " ") {
		strMap[word] = 1
	}

	for index, str := range tfidf.vocabulary {
		_, present := strMap[str]
		if present {
			tfArray[index] = 1
		}
	}
	return tfArray
}

func (tfidf *TfidfVectorizer) Transform(data []string, matrix *mat64.Dense) {
	tfMatrix := mat64.NewDense(len(data), tfidf.size, make([]float64, len(data)*tfidf.size))
	for index, str := range data {
		tfRow := tfidf.calculateTF(str)
		tfMatrix.SetRow(index, tfRow)
	}

	matrix.Mul(tfMatrix, tfidf.idf)
}
