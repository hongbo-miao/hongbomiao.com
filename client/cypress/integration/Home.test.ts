import meQuery from '../../src/Home/queries/me.query';

describe('Home', () => {
  it('render Home', () => {
    cy.visit('https://localhost:8080/');
    cy.contains('Making magic happen');
  });

  it('receive GraphQL meQuery data', () => {
    // eslint-disable-next-line jest/valid-expect-in-promise
    cy.request('POST', 'https://localhost:5000/graphql', {
      query: meQuery,
    }).then((res) => {
      expect(res).property('status').to.equal(200);
      expect(res)
        .property('body')
        .to.eql({
          data: {
            me: {
              name: 'Hongbo Miao',
              slogan: 'Making magic happen',
            },
          },
        });
    });
  });
});
