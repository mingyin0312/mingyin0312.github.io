---
layout: page
permalink: /publications/
title: Publications
description: Publications by categories in reversed chronological order. Also see this <a href='https://scholar.google.com/citations?user=ncBRYIUAAAAJ&hl=en'>Google Scholar</a> page or this <a href='https://www.semanticscholar.org/author/Ming-Yin/2053888252'>Semantic Scholar</a> page. The * denotes equal contribution.
years_pre: [2024]
years_pub: [2024,2023,2022,2021,2020]
years_ws: [2023,2022]
nav: true
---

<br />







### **Preprints**



<div class="publications">

{% for y in page.years_pre %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}},abbr=Preprint] %}
{% endfor %}

</div>



*******
<br />





### **Publications**



<div class="publications">

{% for y in page.years_pub %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}},abbr!~-WS|Preprint] %}
{% endfor %}

</div>


### **Workshop Papers**



<div class="publications">

{% for y in page.years_ws %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}},abbr=ICML-WS] %}
  {% bibliography -f papers -q @*[year={{y}}, abbr=NeurIPS-WS] %}
{% endfor %}

</div>

