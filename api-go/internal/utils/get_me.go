package utils

type Me struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	FirstName string `json:"firstName"`
	LastName  string `json:"lastName"`
	Bio       string `json:"bio"`
}

func GetMe() (me Me, err error) {
	firstName := "Hongbo"
	LastName := "Miao"
	me = Me{
		ID:        "0",
		Name:      firstName + " " + LastName,
		FirstName: firstName,
		LastName:  LastName,
		Bio:       "Making magic happen",
	}
	return me, nil
}
