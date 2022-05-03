// Each node represents an actor
CREATE(:Actor {name: 'Mark Hamill', actor_id: 1}), (:Actor {name: 'Harrison Ford', actor_id: 2}),
      (:Actor {name: 'Carrie Fisher', actor_id: 3})

// This node represent a movie
CREATE(:Movie {title:    'Star Wars: Episode V - The Empire Strikes Back', release_year: 1980,
               movie_id: 1})


// Mark Hamill played Luke Skywalker in Star Wars: Episode V - The Empire Strikes Back'
MATCH (a:Actor), (m:Movie)
  WHERE a.actor_id = 1 AND m.movie_id = 1
CREATE (a)-[r:ACTED_IN {role: 'Luke Skywalker'}]->(m)
RETURN r

// Harrison Ford played Han Solo
MATCH (a:Actor), (m:Movie)
  WHERE a.actor_id = 2 AND m.movie_id = 1
CREATE (a)-[r:ACTED_IN {role: 'Han Solo'}]->(m)
RETURN r

// Carrie Fisher played Princess Leila
MATCH (a:Actor), (m:Movie)
  WHERE a.actor_id = 3 AND m.movie_id = 1
CREATE (a)-[r:ACTED_IN {role: 'Princess Leila'}]->(m)
RETURN r


// Visualize
MATCH (m:Movie)
  WHERE m.movie_id = 1
RETURN m
