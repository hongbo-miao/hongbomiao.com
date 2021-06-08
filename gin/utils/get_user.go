package utils

type User struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

func GetUser(id string) (user User, err error) {
	users := []User{
		{"0", "Hongbo"},
		{"1", "Jack"},
		{"2", "Rose"},
	}
	for _, v := range users {
		if v.ID == id {
			return v, nil
		}
	}
	return
}
