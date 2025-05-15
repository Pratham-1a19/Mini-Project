-- Update the customer preferences directly in the database
-- UPDATE customers
-- SET customer_email = 'pradpat1918@gmail.com'
-- WHERE customer_name = 'Mrudul Dhaduk';

-- PRAGMA table_info(customers);

-- PRAGMA table_info('Baby Products');

-- DELETE FROM 'Baby Products';







UPDATE customers
SET customer_preference = '["Bakery Products", "Dairy Products"]'
WHERE customer_name = 'Mrudul';


