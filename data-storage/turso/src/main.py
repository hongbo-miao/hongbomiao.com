import logging
import uuid
from pathlib import Path

import turso

logger = logging.getLogger(__name__)


def main() -> None:
    database_path = Path("data/users.db")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with turso.connect(str(database_path)) as connection:
        cursor = connection.cursor()

        # Create table
        cursor.execute("""
            create table if not exists users (
                id text primary key,
                name text not null,
                email text unique not null,
                created_at timestamp default current_timestamp
            )
        """)
        connection.commit()
        logger.info("Created users table")

        # Insert data
        users = [
            (str(uuid.uuid4()), "Alice", "alice@example.com"),
            (str(uuid.uuid4()), "Bob", "bob@example.com"),
        ]
        for user in users:
            cursor.execute(
                "insert or ignore into users (id, name, email) values (?, ?, ?)",
                user,
            )
        connection.commit()
        logger.info("Inserted sample users")

        # Query data
        cursor.execute("select id, name, email from users")
        rows = cursor.fetchall()
        logger.info("Users in database:")
        for row in rows:
            logger.info(f"  id={row[0]}, name={row[1]}, email={row[2]}")

        # Update data
        cursor.execute(
            "update users set name = ? where email = ?",
            ("Alice Smith", "alice@example.com"),
        )
        connection.commit()
        logger.info("Updated Alice's name")

        # Query single user
        cursor.execute(
            "select id, name, email from users where email = ?",
            ("alice@example.com",),
        )
        user = cursor.fetchone()
        if user:
            logger.info(f"Updated user: id={user[0]}, name={user[1]}, email={user[2]}")

        # Count users
        cursor.execute("select count(*) as user_count from users")
        result = cursor.fetchone()
        user_count = result[0] if result else 0
        logger.info(f"Total users: {user_count}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
