# Applicability Domain

![python_badge](https://img.shields.io/pypi/pyversions/applicability-domain)
![license_badge](https://img.shields.io/pypi/l/applicability-domain)
![Total_Downloads_badge](https://pepy.tech/badge/applicability-domain)

## How to use.
See example for details.
Now, I' ll show you some easy example.

```python
from applicability_domain import ApplicabilityDomainDetector
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)

ad = ApplicabilityDomainDetector(k = 5, alpha = 0.9)
ad.fit(X_train)

X_test_inlier = ad.transform(X_test)
```

## References
### Sites
* https://datachemeng.com/applicabilitydomain/


## LICENSE
Copyright © 2021 yu9824

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.