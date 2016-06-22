package text

import (
	"testing"

	"github.com/gonum/matrix/mat64"
)

func TfidfSetup() (*TfidfVectorizer, error) {
	vocab := []string{"apples", "bananas", "oranges"}
	data := []string{"apples bananas", "oranges apples", "grapes"}

	tfidf := NewTfidfVectorizer(vocab)
	return tfidf, tfidf.Fit(data)
}

func TestFit(t *testing.T) {
	_, err := TfidfSetup()
	if err != nil {
		t.Errorf("Unable to fit tfidf")
	}
}

func TestTransform(t *testing.T) {
	tfidf, err := TfidfSetup()
	if err != nil {
		t.Errorf("Unable to fit tfidf")
	}

	var matrix mat64.Dense

	newData := []string{"apples grapes", "bananas bananas"}
	tfidf.Transform(newData, &matrix)
}

func TestTransformAndSaveData(t *testing.T) {
	vocab := []string{"apples", "bananas", "oranges"}
	tfidf := NewTfidfVectorizer(vocab)

	var matrix mat64.Dense

	newData := []string{"apples grapes", "bananas bananas", "oranges"}
	labels := []string{"fruits", "fruits", "fruits"}
	saveDir := "result/tfidf_data.csv"

	err := tfidf.FitTransformAndSaveData(newData, labels, &matrix, saveDir)
	if err != nil {
		t.Error(err)
	}
}
