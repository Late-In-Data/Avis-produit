const byId=id=>document.getElementById(id);
const number=value=>value.toLocaleString('fr-FR');
async function jsonFile(path){const response=await fetch(path,{cache:'no-store'});if(!response.ok)throw new Error(path);return response.json();}
async function showTopics(){
  try{
    const report=await jsonFile('topics.json');
    byId('topic-coverage').textContent=`${report.n_topics} thèmes · ${number(report.documents)} avis · ${number(report.unassigned)} sans composante reconnue. Ajustement exploratoire sur l’ensemble de l’échantillon.`;
    const rows=byId('topic-rows');rows.replaceChildren();
    for(const topic of report.topics){const row=document.createElement('tr');for(const value of [topic.label,number(topic.documents),number(topic.positive),number(topic.negative)]){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}rows.append(row);}
    const select=byId('topic-select');select.replaceChildren();
    for(const topic of report.topics){const option=document.createElement('option');option.value=String(topic.id);option.textContent=topic.label;select.append(option);}
    const update=()=>{const topic=report.topics.find(t=>String(t.id)===select.value);byId('topic-details').textContent=`Mots dominants : ${topic.keywords.join(', ')}. Avis à 3 étoiles dans ce thème : ${number(topic.three_stars)}.`;};
    select.disabled=false;select.addEventListener('change',update);update();
  }catch{byId('topic-coverage').textContent='Les thèmes ne sont pas disponibles pour cette version du site.';}
}
async function showRefresh(){
  try{
    const report=await jsonFile('refresh.json');
    const date=new Intl.DateTimeFormat('fr-FR',{timeZone:'Europe/Paris',dateStyle:'medium',timeStyle:'short'}).format(new Date(report.processed_at));
    byId('refresh-date').textContent=`Dernier traitement terminé : ${date}, heure de Paris. Dernier avis Amazon : ${report.amazon.latest_review_at.slice(0,10)}.`;
    const status={not_configured:'Reddit : identifiants non configurés — aucune collecte effectuée.',approval_required:'Reddit : accès à approuver — aucune collecte effectuée.',disabled:'Reddit : collecte désactivée.',error:'Reddit : échec de la collecte — données non actualisées.',ok:'Reddit : collecte terminée.'};
    byId('reddit-status').textContent=status[report.reddit.status]??'Reddit : état inconnu.';
    if(report.reddit.status==='ok'){
      const r=report.reddit;const target=byId('reddit-aggregates');target.hidden=false;
      target.textContent=`${number(r.documents)} discussions collectées ; ${number(r.predicted_positive)} prédites positives, ${number(r.predicted_negative)} prédites négatives. Ces prédictions n’ont pas été validées sur Reddit.`;
    }
  }catch{byId('refresh-date').textContent='La date du dernier traitement n’est pas disponible.';byId('reddit-status').textContent='Le statut Reddit ne peut pas être vérifié.';}
}
showTopics();showRefresh();
