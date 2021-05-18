let super_heros = ['superman', 'batman', 'spiderman', 'ironman', 'captain america']
const ages = {
  age1: 32,
  age2: 33,
  age3: 22,
  age4: 12
};
for (super_hero of super_heros) {
  if (super_hero == 'spiderman') {
    console.log(`coolest character of them all is :: ${super_hero}`);
    break;
  }
}

