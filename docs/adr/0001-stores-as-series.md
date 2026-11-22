# Stores stand in for the paper's products; only the 934 Complete Stores are modelled

The paper forecasts 500 store-item series jointly, but Rossmann has no item level, so each store is one Series and the models predict all stores' next-day sales in one output vector. The 180 stores missing six months of 2014 (and the one store missing a single day) are dropped instead of imputed, so the daily matrix is complete without invented values; this leaves 934 Series instead of the full 1,115.
