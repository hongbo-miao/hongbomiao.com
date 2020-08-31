interface PostgresInputUser {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  bio: string | null | undefined;
}

export default PostgresInputUser;
