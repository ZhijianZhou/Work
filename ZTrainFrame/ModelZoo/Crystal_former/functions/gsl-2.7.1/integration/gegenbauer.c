/* integration/gegenbauer.c
 * 
 * Copyright (C) 2017 Konrad Griessinger, Patrick Alken
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/*
 * The code in this module is based on IQPACK, specifically the LGPL
 * implementation found in HERMITE_RULE:
 * https://people.sc.fsu.edu/~jburkardt/c_src/hermite_rule/hermite_rule.html
 */

#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_gamma.h>

static int
gegenbauer_check(const size_t n, const gsl_integration_fixed_params * params)
{
  (void) n;

  if (fabs(params->b - params->a) <= GSL_DBL_EPSILON)
    {
      GSL_ERROR("|b - a| too small", GSL_EDOM);
    }
  else if (params->a >= params->b)
    {
      GSL_ERROR("lower integration limit must be smaller than upper limit", GSL_EDOM);
    }
  else if (params->alpha <= -1.0)
    {
      GSL_ERROR("alpha must be > -1", GSL_EDOM);
    }
  else
    {
      return GSL_SUCCESS;
    }
}

static int
gegenbauer_init(const size_t n, double * diag, double * subdiag, gsl_integration_fixed_params * params)
{
  size_t i;

  /* construct the diagonal and subdiagonal elements of Jacobi matrix */
  diag[0] = 0.0;
  subdiag[0] = sqrt ( 1.0 / (2.0*params->alpha + 3.0) );
  for (i = 1; i < n; i++)
    {
      diag[i] = 0.0;
      subdiag[i] = sqrt ( (i + 1.0) * (2.0*params->alpha + i + 1.0) / ( 4.0*(params->alpha + i + 1.0)*(params->alpha + i + 1.0) - 1.0 ) );
    }

  params->zemu = M_SQRTPI * gsl_sf_gamma(params->alpha + 1.0) / gsl_sf_gamma(params->alpha + 1.5);
  params->shft = 0.5*(params->b + params->a);
  params->slp = 0.5*(params->b - params->a);
  params->al = params->alpha;
  params->be = params->alpha;

  return GSL_SUCCESS;
}

static const gsl_integration_fixed_type gegenbauer_type =
{
  gegenbauer_check,
  gegenbauer_init
};

const gsl_integration_fixed_type *gsl_integration_fixed_gegenbauer = &gegenbauer_type;
