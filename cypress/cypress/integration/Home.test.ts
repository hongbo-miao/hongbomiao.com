// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore
import { skipOn } from '@cypress/skip-test';
import meQuery from '../../../web/src/Home/queries/me.query';

describe('Home', () => {
  const { ci, domain, serverPort, webPort } = Cypress.env();

  it('render Home', () => {
    cy.visit(`${domain}:${webPort}`);
    cy.contains('Making magic happen');
  });

  skipOn(ci, () => {
    it('receive GraphQL meQuery data', () => {
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
                firstName: 'Hongbo',
                lastName: 'Miao',
                bio: 'Making magic happen',
              },
            },
          });
      });
    });
  });
});
