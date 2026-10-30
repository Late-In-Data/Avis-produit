import {predict} from './model.mjs';
import './topics.mjs';
const examples = {
  positive: 'This coffee tastes wonderful. It arrived fresh and I will definitely buy it again.',
  negative: 'This food tastes horrible and stale. A complete waste of money. I will not buy it again.',
  mixed: 'The chocolate tastes delicious, but the package arrived broken and the price is far too high.'
};
const byId = id => document.getElementById(id);
const form = byId('sentiment-form'), output = byId('prediction');
const percent = n => new Intl.NumberFormat('fr-FR',{style:'percent',maximumFractionDigits:1}).format(n);
const decimal = n => new Intl.NumberFormat('fr-FR',{maximumFractionDigits:3,minimumFractionDigits:3}).format(n);
let modelPromise;
async function getModel(){
  if (!modelPromise) modelPromise = fetch('model.json').then(r=>{if(!r.ok) throw new Error('model');return r.json()}).catch(e=>{modelPromise=null;throw e});
  return modelPromise;
}
function cleanText(text){
  const decoder = document.createElement('textarea');
  decoder.innerHTML = text;
  const decoded = decoder.value;
  return decoded.replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').trim();
}
for (const button of document.querySelectorAll('[data-example]')) button.addEventListener('click',()=>{
  byId('review').value=examples[button.dataset.example]; output.hidden=true; byId('review').focus();
});
form.addEventListener('submit', async event=>{
  event.preventDefault();
  output.replaceChildren(); output.hidden=false;
  const text=cleanText(byId('review').value);
  if((text.match(/[a-zA-Z]+/g)??[]).length<3){output.textContent='Saisissez au moins trois mots en anglais.';return;}
  const button=byId('analyze');button.disabled=true;output.textContent='Chargement du modèle local…';
  try{
    const result=predict(await getModel(),text);output.replaceChildren();
    if(!result.recognized){output.textContent='Aucun terme reconnu. Essayez un avis alimentaire en anglais.';return;}
    const title=document.createElement('strong');title.textContent=result.label;title.className=result.label==='Négatif'?'negative':'';
    const score=document.createElement('p');score.textContent=`Score du modèle pour « positif » : ${percent(result.positive)}. Score non calibré.`;
    output.append(title,score);
    if(result.positive>=.35&&result.positive<=.65){const note=document.createElement('p');note.textContent='Score proche du seuil de 50 % : une relecture est particulièrement utile.';output.append(note);}
  }catch{output.textContent='Le modèle n’a pas pu être chargé. Vérifiez votre connexion puis réessayez.';}
  finally{button.disabled=false;}
});
async function renderResults(){
  try{
    const response=await fetch('results.json');if(!response.ok) throw new Error('results');const data=await response.json();
    const rows=[['Toujours positif · test initial',data.initial.split_counts.test,data.initial.baseline],['TF-IDF · test initial',data.initial.split_counts.test,data.initial.test],['TF-IDF · période postérieure',data.robustness.temporal.counts.test,data.robustness.temporal.test],['TF-IDF · produits inconnus',data.robustness.unseen_products.counts.test,data.robustness.unseen_products.test]];
    if(data.transformer) rows.push(['DistilBERT SST-2 · test initial',data.transformer.test_rows,data.transformer.transformer.metrics]);
    const body=byId('evaluation-rows');body.replaceChildren();
    for(const [name,count,metrics] of rows){const tr=document.createElement('tr');for(const value of [name,count.toLocaleString('fr-FR'),percent(metrics.accuracy),decimal(metrics.macro_f1)]){const td=document.createElement('td');td.textContent=value;tr.append(td)}body.append(tr)}
    if(data.transformer){const b=data.transformer;const target=byId('transformer-result');target.hidden=false;const h=document.createElement('h3');h.textContent='Un transformer ne remplace pas une comparaison.';const p=document.createElement('p');p.textContent=`DistilBERT SST-2 : macro-F1 ${decimal(b.transformer.metrics.macro_f1)} ; TF-IDF : ${decimal(b.logistic.metrics.macro_f1)}. Débit mesuré sur CPU : ${b.transformer.milliseconds_per_review.toFixed(1)} ms/avis contre ${b.logistic.milliseconds_per_review.toFixed(2)} ms/avis pour TF-IDF.`;const caveat=document.createElement('p');caveat.className='small';caveat.textContent=`Lots de ${b.batch_size} pour DistilBERT, après échauffement et tokenisation incluse. ${b.truncated_reviews} avis tronqués à ${b.max_length} tokens. Modèle préentraîné sur SST-2, sans adaptation Amazon. Cette mesure n’est pas une latence de requête isolée.`;target.append(h,p,caveat);}
  }catch{byId('evaluation-rows').innerHTML='<tr><td colspan="4">Les résultats ne sont pas disponibles. Consultez les rapports du dépôt.</td></tr>';}
}
renderResults();
