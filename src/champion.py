"""Regla para decidir si un modelo nuevo reemplaza al champion.

Es lógica pura (sin MLflow) para poder probarla con pytest.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    promover: bool
    motivo: str


def decidir_promocion(
    metricas_nuevo: dict[str, float],
    metrica_champion: float | None,
    metrica: str = "r2",
    umbral_minimo: float = 0.90,
    mejora_minima: float = 0.0,
    mae_maximo: float | None = None,
) -> Decision:
    """El modelo nuevo pasa a champion solo si cumple las tres condiciones:

    1. Su métrica supera el umbral mínimo.
    2. Su error medio (MAE) no pasa del máximo aceptado (criterio adicional del grupo).
    3. Supera al champion vigente por al menos `mejora_minima` (si ya hay champion).
    """
    valor = metricas_nuevo[metrica]
    if valor < umbral_minimo:
        return Decision(False, f"{metrica}={valor:.4f} no alcanza el mínimo {umbral_minimo}")
    if mae_maximo is not None and metricas_nuevo["mae"] > mae_maximo:
        return Decision(False, f"mae={metricas_nuevo['mae']:.2f} supera el máximo aceptado {mae_maximo}")
    if metrica_champion is None:
        return Decision(True, f"{metrica}={valor:.4f} supera el mínimo y todavía no hay champion")
    if valor < metrica_champion + mejora_minima:
        return Decision(False, f"{metrica}={valor:.4f} no mejora al champion ({metrica_champion:.4f}) "
                               f"en al menos {mejora_minima}")
    return Decision(True, f"{metrica}={valor:.4f} mejora al champion ({metrica_champion:.4f})")
