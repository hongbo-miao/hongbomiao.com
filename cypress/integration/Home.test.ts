import meQuery from '../../client/src/Home/queries/me.query';

describe('Home', () => {
  const { domain, clientPort, serverPort } = Cypress.env();

  it('render Home', () => {
    cy.visit(`${domain}:${clientPort}`);
    cy.contains('Making magic happen');
  });

  it('receive GraphQL meQuery data', () => {
    // eslint-disable-next-line jest/valid-expect-in-promise
    cy.request('POST', `${domain}:${serverPort}/graphql`, {
      query: meQuery,
    }).then((res) => {
      expect(res).property('status').to.equal(200);
      expect(res)
        .property('body')
        .to.eql({
          data: {
            me: {
              name: 'Hongbo Miao',
              bio: 'Making magic happen',
            },
          },
        });
    });
  });
});
