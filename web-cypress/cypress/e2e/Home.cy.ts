describe('Home', () => {
  const { domain, serverPort, webPort } = Cypress.env();

  it('render Home', () => {
    cy.visit(`${domain}:${webPort}`);
    cy.get('.hm-home-container').contains('Making magic happen');
  });

  it('receive GraphQL meQuery data', () => {
    const query = `
      query Me {
        me {
          name
          bio
        }
      }
    `;

    cy.request('POST', `${domain}:${serverPort}/graphql`, {
      query,
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
