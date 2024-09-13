echo "----------Start unittest----------"
coverage run -m unittest
echo "----------End unittest----------"

echo "----------Code coverage----------"
coverage report -m
