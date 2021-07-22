import bcrypt from 'bcrypt';
import jsonwebtoken from 'jsonwebtoken';
import config from '../../config';
import findUserByEmail from '../../dataSources/postgres/utils/findUserByEmail';

const getJWTToken = async (email: string, password: string): Promise<string> => {
  const user = await findUserByEmail(email);
  if (user == null) {
    throw new Error('No user with this email.');
  }

  const isPasswordValid = await bcrypt.compare(password, user.password);
  if (!isPasswordValid) {
    throw new Error('Incorrect password.');
  }

  return jsonwebtoken.sign({ id: user.id, email: user.email }, config.jwtSecret, { expiresIn: '1d' });
};

export default getJWTToken;
