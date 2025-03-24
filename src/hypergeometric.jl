### adapted from:
#template <bool calc_a1, bool calc_a2, bool calc_b1, typename T1, typename T2,
#          typename T3, typename T_z,
#          typename ScalarT = return_type_t<T1, T2, T3, T_z>,
#          typename TupleT = std::tuple<ScalarT, ScalarT, ScalarT>>
#TupleT grad_2F1_impl_ab(const T1& a1, const T2& a2, const T3& b1, const T_z& z,
#                        double precision = 1e-14, int max_steps = 1e6) {
#  TupleT grad_tuple = TupleT(0, 0, 0);
#
#  if (z == 0) {
#    return grad_tuple;
#  }
#
#  using ScalarArrayT = Eigen::Array<ScalarT, 3, 1>;
#  ScalarArrayT log_g_old = ScalarArrayT::Constant(3, 1, NEGATIVE_INFTY);
#
#  ScalarT log_t_old = 0.0;
#  ScalarT log_t_new = 0.0;
#  int sign_z = sign(z);
#  auto log_z = log(abs(z));
#
#  int log_t_new_sign = 1.0;
#  int log_t_old_sign = 1.0;
#
#  Eigen::Array<int, 3, 1> log_g_old_sign = Eigen::Array<int, 3, 1>::Ones(3);
#
#  int sign_zk = sign_z;
#  int k = 0;
#  const int min_steps = 5;
#  ScalarT inner_diff = 1;
#  ScalarArrayT g_current = ScalarArrayT::Zero(3);
#
#  while ((inner_diff > precision || k < min_steps) && k < max_steps) {
#    ScalarT p = ((a1 + k) * (a2 + k) / ((b1 + k) * (1.0 + k)));
#    if (p == 0) {
#      return grad_tuple;
#    }
#    log_t_new += log(fabs(p)) + log_z;
#    log_t_new_sign = sign(value_of_rec(p)) * log_t_new_sign;
#
#    if (calc_a1) {
#      ScalarT term_a1
#          = log_g_old_sign(0) * log_t_old_sign * exp(log_g_old(0) - log_t_old)
#            + inv(a1 + k);
#      log_g_old(0) = log_t_new + log(abs(term_a1));
#      log_g_old_sign(0) = sign(value_of_rec(term_a1)) * log_t_new_sign;
#      g_current(0) = log_g_old_sign(0) * exp(log_g_old(0)) * sign_zk;
#      std::get<0>(grad_tuple) += g_current(0);
#    }
#
#    if (calc_a2) {
#      ScalarT term_a2
#          = log_g_old_sign(1) * log_t_old_sign * exp(log_g_old(1) - log_t_old)
#            + inv(a2 + k);
#      log_g_old(1) = log_t_new + log(abs(term_a2));
#      log_g_old_sign(1) = sign(value_of_rec(term_a2)) * log_t_new_sign;
#      g_current(1) = log_g_old_sign(1) * exp(log_g_old(1)) * sign_zk;
#      std::get<1>(grad_tuple) += g_current(1);
#    }
#
#    if (calc_b1) {
#      ScalarT term_b1
#          = log_g_old_sign(2) * log_t_old_sign * exp(log_g_old(2) - log_t_old)
#            + inv(-(b1 + k));
#      log_g_old(2) = log_t_new + log(abs(term_b1));
#      log_g_old_sign(2) = sign(value_of_rec(term_b1)) * log_t_new_sign;
#      g_current(2) = log_g_old_sign(2) * exp(log_g_old(2)) * sign_zk;
#      std::get<2>(grad_tuple) += g_current(2);
#    }
#
#    inner_diff = g_current.array().abs().maxCoeff();
#
#    log_t_old = log_t_new;
#    log_t_old_sign = log_t_new_sign;
#    sign_zk *= sign_z;
#    ++k;
#  }
#
#  if (k > max_steps) {
#    throw_domain_error("grad_2F1", "k (internal counter)", max_steps,
#                       "exceeded ",
#                       " iterations, hypergeometric function gradient "
#                       "did not converge.");
#  }
#  return grad_tuple;
#}
#
#template <bool calc_a1, bool calc_a2, bool calc_b1, bool calc_z, typename T1,
#          typename T2, typename T3, typename T_z,
#          typename ScalarT = return_type_t<T1, T2, T3, T_z>,
#          typename TupleT = std::tuple<ScalarT, ScalarT, ScalarT, ScalarT>>
#TupleT grad_2F1_impl(const T1& a1, const T2& a2, const T3& b1, const T_z& z,
#                     double precision = 1e-14, int max_steps = 1e6) {
#  bool euler_transform = false;
#  try {
#    check_2F1_converges("hypergeometric_2F1", a1, a2, b1, z);
#  } catch (const std::exception& e) {
#    // Apply Euler's hypergeometric transformation if function
#    // will not converge with current arguments
#    check_2F1_converges("hypergeometric_2F1 (euler transform)", b1 - a1, a2, b1,
#                        z / (z - 1));
#    euler_transform = true;
#  }
#
#  std::tuple<ScalarT, ScalarT, ScalarT> grad_tuple_ab;
#  TupleT grad_tuple_rtn = TupleT(0, 0, 0, 0);
#  if (euler_transform) {
#    ScalarT a1_euler = a2;
#    ScalarT a2_euler = b1 - a1;
#    ScalarT z_euler = z / (z - 1);
#    if (calc_z) {
#      auto hyper1 = hypergeometric_2F1(a1_euler, a2_euler, b1, z_euler);
#      auto hyper2 = hypergeometric_2F1(1 + a2, 1 - a1 + b1, 1 + b1, z_euler);
#      std::get<3>(grad_tuple_rtn)
#          = a2 * pow(1 - z, -1 - a2) * hyper1
#            + (a2 * (b1 - a1) * pow(1 - z, -a2)
#               * (inv(z - 1) - z / square(z - 1)) * hyper2)
#                  / b1;
#    }
#    if (calc_a1 || calc_a2 || calc_b1) {
#      // 'a' gradients under Euler transform are constructed using the gradients
#      // of both elements, so need to compute both if any are required
#      constexpr bool calc_a1_euler = calc_a1 || calc_a2;
#      // 'b' gradients under Euler transform require gradients from 'a2'
#      constexpr bool calc_a2_euler = calc_a1 || calc_a2 || calc_b1;
#      grad_tuple_ab = grad_2F1_impl_ab<calc_a1_euler, calc_a2_euler, calc_b1>(
#          a1_euler, a2_euler, b1, z_euler);
#
#      auto pre_mult_ab = inv(pow(1.0 - z, a2));
#      if (calc_a1) {
#        std::get<0>(grad_tuple_rtn) = -pre_mult_ab * std::get<1>(grad_tuple_ab);
#      }
#      if (calc_a2) {
#        auto hyper_da2 = hypergeometric_2F1(a1_euler, a2, b1, z_euler);
#        std::get<1>(grad_tuple_rtn)
#            = -pre_mult_ab * hyper_da2 * log1m(z)
#              + pre_mult_ab * std::get<0>(grad_tuple_ab);
#      }
#      if (calc_b1) {
#        std::get<2>(grad_tuple_rtn)
#            = pre_mult_ab
#              * (std::get<1>(grad_tuple_ab) + std::get<2>(grad_tuple_ab));
#      }
#    }
#  } else {
#    if (calc_z) {
#      auto hyper_2f1_dz = hypergeometric_2F1(a1 + 1.0, a2 + 1.0, b1 + 1.0, z);
#      std::get<3>(grad_tuple_rtn) = (a1 * a2 * hyper_2f1_dz) / b1;
#    }
#    if (calc_a1 || calc_a2 || calc_b1) {
#      grad_tuple_ab
#          = grad_2F1_impl_ab<calc_a1, calc_a2, calc_b1>(a1, a2, b1, z);
#      if (calc_a1) {
#        std::get<0>(grad_tuple_rtn) = std::get<0>(grad_tuple_ab);
#      }
#      if (calc_a2) {
#        std::get<1>(grad_tuple_rtn) = std::get<1>(grad_tuple_ab);
#      }
#      if (calc_b1) {
#        std::get<2>(grad_tuple_rtn) = std::get<2>(grad_tuple_ab);
#      }
#    }
#  }
#  return grad_tuple_rtn;
#}

using Arblib: ArbLike

function grad_2F1_impl_ab(_a1, _a2, _b1, _z, precision = 1.0e-14, max_steps = 10^6; prec)
    grad = [Arb(0; prec) for _ in 1:3]
    if iszero(_z)
        return grad
    end

    a1 = ForwardDiff.value(_a1)
    a2 = ForwardDiff.value(_a2)
    b1 = ForwardDiff.value(_b1)
    z = ForwardDiff.value(_z)

    log_g_old = [Arb(-Inf; prec) for _ in 1:3]
    log_t_old = Arb(0; prec)
    log_t_new = Arb(0; prec)
    sign_z = sign(z)
    log_z = log(abs(z))

    log_t_new_sign = 1
    log_t_old_sign = 1

    log_g_old_sign = [1 for _ in 1:3]

    sign_zk = sign_z
    k = 0
    min_steps = 5
    inner_diff = Arb(1; prec)
    g_current = [Arb(0; prec) for _ in 1:3]

    while (inner_diff > precision || k < min_steps) && k < max_steps
        p = (a1 + k) * (a2 + k) / ((b1 + k) * (1 + k))
        if min(a1, a2) == -k
            return grad
        end
        log_t_new += log(abs(p)) + log_z
        log_t_new_sign = sign(p) * log_t_new_sign * sign_z

        if _a1 isa Dual
            term_a1 = log_g_old_sign[1] * log_t_old_sign * exp(log_g_old[1] - log_t_old) + inv(a1 + k)
            if iszero(p)
                p′ = (a2 + k) / ((b1 + k) * (1 + k))
                log_g_old[1] = log_t_old + log(abs(p′)) + log_z
                log_g_old_sign[1] = log_t_old_sign * sign(p′) * sign_z
            elseif !isfinite(log_t_new)
                log_g_old[1] += log(abs(p)) + log_z
                log_g_old_sign[1] = log_g_old_sign * sign(p) * sign_z
            else
                log_g_old[1] = log_t_new + log(abs(term_a1))
                log_g_old_sign[1] = sign(term_a1) * log_t_new_sign
            end
            g_current[1] = log_g_old_sign[1] * exp(log_g_old[1]) * sign_zk
            grad[1] += g_current[1]
        end

        if _a2 isa Dual
            term_a2 = log_g_old_sign[2] * log_t_old_sign * exp(log_g_old[2] - log_t_old) + inv(a2 + k)
            if iszero(p)
                p′ = (a1 + k) / ((b1 + k) * (1 + k))
                log_g_old[2] = log_t_old + log(abs(p′)) + log_z
                log_g_old_sign[2] = log_t_old_sign * sign(p′) * sign_z
            elseif !isfinite(log_t_new)
                log_g_old[2] += log(abs(p)) + log_z
                log_g_old_sign[2] *= sign(p) * sign_z
            else
                log_g_old[2] = log_t_new + log(abs(term_a2))
                log_g_old_sign[2] = sign(term_a2) * log_t_new_sign
            end
            g_current[2] = log_g_old_sign[2] * exp(log_g_old[2]) * sign_zk
            grad[2] += g_current[2]
        end

        if _b1 isa Dual
            term_b1 = log_g_old_sign[3] * log_t_old_sign * exp(log_g_old[3] - log_t_old) + inv(-(b1 + k))
            log_g_old[3] = log_t_new + log(abs(term_b1))
            log_g_old_sign[3] = sign(term_b1) * log_t_new_sign
            g_current[3] = log_g_old_sign[3] * exp(log_g_old[3]) * sign_zk
            grad[3] += g_current[3]
        end

        inner_diff = maximum(abs, g_current)

        log_t_old = log_t_new
        log_t_old_sign = log_t_new_sign
        sign_zk *= sign_z
        k += 1
    end

    if k == max_steps
        throw(DomainError(max_steps, "k (internal counter) $max_steps exceeded iterations, hypergeometric function gradient did not converge."))
    end

    return grad
end

function grad_2F1_impl(_a1, _a2, _b1, _z, precision = 1.0e-14, max_steps = 10^6; prec)
    grad_rtn = [Arb(0; prec) for _ in 1:4]

    a1 = ForwardDiff.value(_a1)
    a2 = ForwardDiff.value(_a2)
    b1 = ForwardDiff.value(_b1)
    z = ForwardDiff.value(_z)

    if _z isa Dual
        hyper_2f1_dz = Arblib.hypgeom_2f1!(Arb(; prec), a1 + 1, a2 + 1, b1 + 1, z, 0)
        grad_rtn[4] = (a1 * a2 * hyper_2f1_dz) / b1
    end
    if _a1 isa Dual || _a2 isa Dual || _b1 isa Dual
        grad_ab = grad_2F1_impl_ab(_a1, _a2, _b1, _z, precision, max_steps; prec)
        if _a1 isa Dual
            grad_rtn[1] = grad_ab[1]
        end
        if _a2 isa Dual
            grad_rtn[2] = grad_ab[2]
        end
        if _b1 isa Dual
            grad_rtn[3] = grad_ab[3]
        end
    end
    return grad_rtn
end

MaybeDualArbLike = Union{ArbLike, Dual{<:Any, <:ArbLike}}

function Arblib.hypgeom_2f1!(res::ArbLike, a::MaybeDualArbLike, b::MaybeDualArbLike, c::MaybeDualArbLike, z::MaybeDualArbLike, regularized::Integer; prec = Arblib._precision(res))
    @assert regularized == 0

    tag = ForwardDiff.tagtype(a)
    tag′ = ForwardDiff.tagtype(b)
    if tag′ !== Nothing
        if tag !== Nothing
            @assert tag == tag′
        end
        tag = tag′
    end
    tag′ = ForwardDiff.tagtype(c)
    if tag′ !== Nothing
        if tag !== Nothing
            @assert tag == tag′
        end
        tag = tag′
    end
    tag′ = ForwardDiff.tagtype(z)
    if tag !== Nothing && tag′ !== Nothing
        @assert tag == tag′
        tag = tag′
    end

    grad = grad_2F1_impl(a, b, c, z; prec)
    partial = ForwardDiff.Partials{0, Arb}(())
    if a isa Dual
        partial += grad[1] * ForwardDiff.partials(a)
        a = ForwardDiff.value(a)
    end
    if b isa Dual
        partial += grad[2] * ForwardDiff.partials(b)
        b = ForwardDiff.value(b)
    end
    if c isa Dual
        partial += grad[3] * ForwardDiff.partials(c)
        c = ForwardDiff.value(c)
    end
    if z isa Dual
        partial += grad[4] * ForwardDiff.partials(z)
        z = ForwardDiff.value(z)
    end

    res = Arblib.hypgeom_2f1!(res, a, b, c, z, 0)
    return Dual{tag}(res, partial)
end

function Arblib.hypgeom_rising!(res::ArbLike, x::Dual{<:Any, <:ArbLike}, n::ArbLike; prec = Arblib._precision(res))
    @assert isinteger(n)
    return prod(i -> x + i, 0:Int(n - 1))
end
