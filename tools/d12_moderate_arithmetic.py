"""Bounded local-kernel comparisons for report reconstruction, never a D12 solve."""


def check_arithmetic(jax, *, probe=None):
    import numpy as np
    import jax.numpy as jnp
    from cffg import hybrid_natural as hn, hybrid_separator as hs
    from cffg.hybrid_joint import covariance_difference
    from d12_compile_fixture import fixture_arrays
    from d12_compile_scaling import checked_arrays
    from d12_moderate_payload import close
    from d12_moderate_reconstruct import affine, learning, project

    parameters, _ = fixture_arrays(32)
    parameters = {name: value[:1] for name, value in parameters.items()}
    weights, offsets = (
        jnp.asarray(parameters[name][0]) for name in ("weights", "offsets")
    )
    center = np.broadcast_to(np.float32([32, -16, 8, -4]), (4, 4, 4)).copy()
    displacement = (np.arange(64, dtype=np.float32).reshape(4, 4, 4) - 32) / 16
    high = np.broadcast_to(np.eye(4, dtype=np.float32) * 1e5, (4, 4, 4, 4)).copy()
    low = high * (np.finfo(np.float32).eps / 4)

    @jax.jit
    def kernels(probability, displacement, high, low, center, xi):
        region = hn.from_centered_moments(
            probability, displacement, high, center, covariance_correction=low
        )
        auxiliary = hs.auxiliary_statistics(region, weights, offsets, xi)
        left, right = (hn.marginal(region, side) for side in (0, 1))
        difference = covariance_difference(
            region, region._replace(covariance_correction=jnp.zeros_like(low))
        )
        return (
            region,
            auxiliary,
            left,
            right,
            auxiliary.physical_linear,
            auxiliary.physical_precision,
            difference,
        )

    retained = None
    for sparse in (False, True):
        probability = (
            np.arange(1, 17, dtype=np.float32).reshape(4, 4)
            if probe == "weighted"
            else np.ones((4, 4), np.float32)
        )
        if sparse:
            probability[0] = 0
            probability[1] *= np.float32(1e-12 if probe else 0)
        probability /= probability.sum()
        xi = np.full((4, 3), int(sparse), np.float32)
        arguments = tuple(
            jnp.asarray(a) for a in (probability, displacement, high, low, center, xi)
        )
        checked_arrays(arguments)
        actual = jax.block_until_ready(kernels(*arguments))
        checked_arrays(actual)
        region, auxiliary, left, right, linear, precision, difference = actual
        assert (
            int(region.status)
            == int(auxiliary.status)
            == int(left.status)
            == int(right.status)
            == 0
        )
        saved = {
            name: np.asarray(value)[None] for name, value in region._asdict().items()
        }
        expected = learning(saved, parameters, xi[None])
        for name in ("target", "residual", "exposure", "linear", "precision", "center"):
            close(np.asarray(getattr(auxiliary, name)), expected[name][0], name)
        close(linear, expected["physical_linear"][0], "physical linear")
        close(precision, expected["physical_precision"][0], "physical precision")
        for side, native in enumerate((left, right)):
            expected = project(saved, side)
            for name, value in expected.items():
                actual_value = np.asarray(getattr(native, name))
                try:
                    close(actual_value, value[0], "projection " + name)
                except ValueError as error:
                    raise ValueError(
                        f"projection {name}, sparse={sparse}, side={side}: "
                        f"native={actual_value.tolist()}, replay={value[0].tolist()}"
                    ) from error
        rounded = np.asarray(region.full_covariance) - np.asarray(region.covariance)
        assert np.all(rounded == 0)
        retained = float(np.max(np.asarray(difference)))
        assert retained > 0
        replay = affine(
            np.ones_like(high)[..., None],
            np.asarray(region.covariance)[..., None],
            -np.asarray(region.covariance),
            np.asarray(region.correction),
        )
        close(difference, replay, "retained covariance difference")
        # The covariance word survives residual subtraction, but the native
        # auxiliary rule consumes full_covariance, rounded to one binary32 word.
        without_low = dict(
            saved, covariance_correction=np.zeros_like(saved["covariance_correction"])
        )
        rounded_learning = learning(without_low, parameters, xi[None])
        for name in ("target", "linear", "precision"):
            np.testing.assert_array_equal(
                learning(saved, parameters, xi[None])[name], rounded_learning[name]
            )
    return dict(
        native=not jax.config.x64_enabled,
        local_cases=2,
        complete_program_calls=0,
        retained_low_word=retained,
    )
