describe('Home', () => {
  it('render Home', () => {
    cy.visit('https://localhost:3000/');
    cy.contains('Making magic happen');
  });
});
