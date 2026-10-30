// Reproduction du TF-IDF (unigrammes/bigrammes, TF sous-linéaire, norme L2)
// et de la régression logistique de scikit-learn. Aucun appel à une API.
export function predict(model, text) {
  const tokens = text.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) ?? [];
  const counts = new Map();
  const add = term => {
    if (Object.hasOwn(model.features, term)) counts.set(term, (counts.get(term) ?? 0) + 1);
  };
  for (const token of tokens) add(token);
  for (let i = 0; i + 1 < tokens.length; i++) add(`${tokens[i]} ${tokens[i + 1]}`);
  let squaredNorm = 0, contribution = 0;
  for (const [term, count] of counts) {
    const [idf, coefficient] = model.features[term];
    const value = (1 + Math.log(count)) * idf;
    squaredNorm += value * value;
    contribution += value * coefficient;
  }
  const decision = model.intercept + (squaredNorm ? contribution / Math.sqrt(squaredNorm) : 0);
  const positive = 1 / (1 + Math.exp(-decision));
  return { label: decision > 0 ? model.classes[1] : model.classes[0], positive, recognized: counts.size };
}
